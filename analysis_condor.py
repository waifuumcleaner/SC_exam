#!/usr/bin/env python
# coding: utf-8

import contextlib
import io
import logging
import os
import pickle
import requests
import sys

import lecroyparser
import numpy as np
import pandas as pd
import ROOT
from scipy import signal, integrate


USAMPLE = 150 # -1: read all (https://pypi.org/project/lecroyparser/)
PMT_NAMES = [f'PMT_{i}' for i in range(1,5)]
PARAM_NAMES = ["a", "c", "tau_s", "tau"]
PARAM_ESTIMATES = [-0.5, -7.75e-8, 9.3e-10, 3.5e-9]
PARAM_BOUNDS = ([-10, +4e-8, 1e-10, 1e-10], [0, +5.5e-8, 1e-8, 1e-8])

def load_pickle_files():
    try:
        with open('./voltages_list.pkl', 'rb') as file:
            voltages_list = pickle.load(file)
        logging.info('voltages_list.pkl successfully loaded')
        with open('./pmt_cuts.pkl', 'rb') as file:
            pmt_cuts = pickle.load(file)
        logging.info('pmt_cuts.pkl successfully loaded')
        return voltages_list, pmt_cuts
    except Exception as e:
        logging.error(f"Error loading .pkl file: {e}")
        sys.exit(1)


def create_fitting_function(b):
    def fitting_function(x, a, c, tau_s, tau):
        if tau == tau_s:
            return np.where(x < c, b, a * (x - c) * np.exp(-(x - c) / tau_s))
        return np.where(x < c, b, -a * (np.exp(-(x - c) / tau_s) - np.exp(-(x - c) / tau)) + b)
    return fitting_function

def create_fitting_function_root(b):
    def fitting_function_root(x, par):
        a, c, tau_s, tau = par[0], par[1], par[2], par[3] 
        if x[0] < c:
            func = b
        else:
            func = -a * (ROOT.TMath.Exp(-(x[0] - c) / tau_s) - ROOT.TMath.Exp(-(x[0] - c) / tau)) + b
            if tau == tau_s:
                func = a * (x[0] - c) * ROOT.TMath.Exp(-(x[0] - c) / tau_s)
        return func
    return fitting_function_root

# Create a context manager to temporarily redirect stdout to /dev/null
@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as null_stream:
        original_stdout = sys.stdout
        sys.stdout = null_stream
        yield
        sys.stdout = original_stdout

def process_files(dir_url, i_pmt, n_files, i_hv, hv, cut, is_noisy_pmt):
    charge = pd.Series(dtype = float, index=range(n_files))
    charge[:] = np.nan
    
    for i_file in range(n_files):
        file_number = f"{i_file:05d}"
        file_name = f"C1--n16deg--{file_number}.trc"
        full_url = f"{dir_url}{file_name}"
        try:
            response = requests.get(full_url)
            response.raise_for_status()  # Raise an exception for non-200 status codes
            f = io.BytesIO(response.content)
            logging.info(f'Analysing {file_name}')
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to open file {full_url}: {e}")
            continue

        data_raw = lecroyparser.ScopeData(data=f.read(), sparse=-1)
        data_raw_rs = signal.resample(data_raw.y, USAMPLE, data_raw.x)
        data = pd.DataFrame({'Time (s)': data_raw_rs[1], 'Amplitude (V)': data_raw_rs[0]})

        min_index = data['Amplitude (V)'].iloc[20:-20].idxmin()
        if pd.isna(min_index) or min_index <= 11 or data['Amplitude (V)'].iloc[min_index] >= cut:
            logging.warning(f"Warning: minimum not found for {PMT_NAMES[i_pmt-1]}, {hv} V, file {file_name}")
            continue

        a = data['Amplitude (V)'].iloc[min_index] * 2.5 
        end_bkg_index = (data['Amplitude (V)'][:min_index][::-1] > 0).idxmax()
        if end_bkg_index <= 10:
            end_bkg_index = 15
        b = data['Amplitude (V)'][1:end_bkg_index].mean()
        c_index = (data['Amplitude (V)'][:min_index][::-1] >= 0.04 * a / 2.5).idxmax()
        c = data['Time (s)'].iloc[c_index]
        tau_s = PARAM_ESTIMATES[2]
        tau = PARAM_ESTIMATES[3]
        initial_guesses = [a, c, tau_s, tau]

        all_parameters_within_bounds = all([lb <= param <= ub for param, lb, ub in zip(initial_guesses, PARAM_BOUNDS[0], PARAM_BOUNDS[1])])
        if not all_parameters_within_bounds:
            for param, (lower_bound, upper_bound), param_name in zip(initial_guesses, zip(*PARAM_BOUNDS), PARAM_NAMES):
                if not (lower_bound <= param <= upper_bound):
                    logging.warning(f"Warning: Parameter {param_name} not between bounds for {PMT_NAMES[i_pmt-1]}, {hv} V, file {file_name}")
            continue

        start_index = c_index - (20 if not is_noisy_pmt else 13)
        end_index = (data['Amplitude (V)'].iloc[min_index:] >= 0.04 * a / 2.5).idxmax() + (30 if not is_noisy_pmt else 20)
        if start_index <= 0 or end_index >= data.shape[0] :
            logging.warning(f"Warning: integration range out of data range, skipping {PMT_NAMES[i_pmt-1]}, {hv} V, file {file_name}")
            continue

        ###### ROOT ######
        graph = ROOT.TGraph(len(data['Time (s)']), np.array(data['Time (s)'], dtype=np.float64), np.array(data['Amplitude (V)'], dtype=np.float64))
        fitting_function_root = create_fitting_function_root(b)
        fit_function_root = ROOT.TF1("fit_function_root", fitting_function_root, data['Time (s)'].iloc[start_index], data['Time (s)'].iloc[end_index], 4)
        fit_function_root.SetParameters(*initial_guesses)
        for i in range(4):
            fit_function_root.SetParLimits(i, PARAM_BOUNDS[0][i], PARAM_BOUNDS[1][i])
        
        with suppress_output():
            graph.Fit(fit_function_root, "RQ")
        fit_results = graph.GetFunction("fit_function_root")
        fitted_parameters = [fit_results.GetParameter(i) for i in range(4)]

        fitting_function = create_fitting_function(b)
        integration_function = lambda x: fitting_function(x, *fitted_parameters)
        charge[i_file], _ = integrate.quad(integration_function, data["Time (s)"].loc[start_index], data["Time (s)"].loc[end_index])

    charge = charge[:i_file + 1]
    mean_charge = charge.mean() / 50. * 1e+12
    std_dev_charge = charge.std() / 50. * 1e+12
    return (i_hv, hv, mean_charge, std_dev_charge)

def main():
    if len(sys.argv) != 5:
        logging.error(f"Usage: python3 analysis_condor.py <dir_url> <i_pmt> <hv> <n_files>, {len(sys.argv)} arguments provided.")
        sys.exit(1)

    dir_url, i_pmt, hv, n_files = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

    voltages_list, pmt_cuts = load_pickle_files()
    hv_list = voltages_list[i_pmt-1]
    i_hv = hv_list.index(str(hv))
    cut = pmt_cuts[i_pmt-1][i_hv]

    charges = pd.DataFrame(columns=['mean_charge', 'std_dev_charge', 'hv'])
    charges[:] = np.nan

    is_noisy_pmt = PMT_NAMES[i_pmt - 1] in ["PMT_2", "PMT_4"]
    try:
        i_hv, hv, mean_charge, std_dev_charge = process_files(dir_url, i_pmt, n_files, i_hv, hv, cut, is_noisy_pmt)
        charges.loc[i_hv] = [mean_charge, std_dev_charge, float(hv)]
    except Exception as e:
        logging.error(f"An error occurred: {e}")

    output_file = f'pmt_data_list.pkl'

    with open(output_file, 'wb') as file:
        pickle.dump(charges, file)

    logging.info("All tasks completed.")

if __name__ == "__main__":
    os.listdir("./")
    main()