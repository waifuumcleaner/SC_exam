#!/usr/bin/env python
# coding: utf-8

import argparse
import contextlib
import logging
import os
import pickle
import requests
import sys

import lecroyparser
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ROOT
from scipy import signal as scipy_signal, integrate
import seaborn as sns

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

matplotlib.use('Agg')
sns.set_theme()

USAMPLE = 150 # -1: read all (https://pypi.org/project/lecroyparser/)
PMT_NAMES = [f'PMT_{i}' for i in range(1,5)]
PARAM_NAMES = ["a", "c", "tau_s", "tau"]
PARAM_ESTIMATES = [-0.5, -7.75e-8, 9.3e-10, 3.5e-9]
PARAM_BOUNDS = ([-10, +4e-8, 1e-10, 1e-10], [0, +5.5e-8, 1e-8, 1e-8])

def list_directories():
    data_dir = "./data/"
    outputs_dir = "./outputs/"
    lists_dir = os.path.join(outputs_dir, "lists")
    logs_dir = os.path.join(outputs_dir, "logs")

    for dir_path in [data_dir, lists_dir, logs_dir]:
        os.makedirs(dir_path, exist_ok=True)

    for i_pmt in range(1, 5): 
        file_path = f"./data_info/data_directories_pmt_{i_pmt}.txt"
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 1:
                        url = parts[0].split('Autosave')[0]
                        relative_path = url.replace('https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/GIN_PMT/', '')
                        folder_path = os.path.join(data_dir, relative_path)
                        os.makedirs(folder_path, exist_ok=True)

    folder_paths = os.listdir(data_dir)
    folder_paths.sort()

    print("\nPMT data folder paths which will be analyzed:\n")
    for i, path in enumerate(folder_paths):
        full_path = os.path.join(data_dir, path)
        if os.path.isdir(full_path):
            print(f"{i+1}.\t{full_path} , \t labeled as PMT_{i+1}")
            voltages = [v for v in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, v))]
            voltages.sort(key=int) 
            print(f"\tHV values for PMT_{i+1} ({len(voltages)} values):\n\t{voltages}\n")

def load_pmt_cuts():
    try:
        with open('./data_info/pmt_cuts.pkl', 'rb') as file:
            pmt_cuts = pickle.load(file)
        print('pmt_cuts.pkl successfully loaded')
        return pmt_cuts
    except Exception as e:
        print(f"Error loading pmt_cuts.pkl: {e}")
        sys.exit(1)

def load_pickle_lists():
    try:
        pmt_data_list = []
        pmt_names = [f'PMT_{i+1}' for i in range(4)]

        for i_pmt in range(1, 5):
            with open(f'outputs/lists/pmt_{i_pmt}_data_list.pkl', 'rb') as file:
                pmt_data_list.append(pickle.load(file))
                print(f'outputs/lists/pmt_{i_pmt}_data_list.pkl successfully loaded')

        return pmt_data_list, pmt_names

    except Exception as e:
        print(f"Error loading pickle file: {e}")
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

def configure_logger(pmt_number, hv):
    logger = logging.getLogger(f"PMT_{pmt_number}_{hv}V_logger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.FileHandler(f'outputs/logs/pmt_{pmt_number}_{hv}V.log', mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def read_parameters(file_path):
    dir_urls, hv_list, n_files_list = [], [], []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    for line in lines:
        dir_url, pmt_number, hv, n_files = line.split()
        pmt_number = int(pmt_number)
        assert pmt_number == int(file_path.split('_')[-1][0]), f"PMT number mismatch in {file_path}: {pmt_number} != {int(file_path.split('_')[-1][0])}"
        hv = int(hv)
        n_files = int(n_files)
        dir_urls.append(dir_url)
        hv_list.append(hv)
        n_files_list.append(n_files)
    return dir_urls, hv_list, n_files_list

def download_file(full_url, full_path, logger, verbose):
    try:
        response = requests.get(full_url)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'wb') as f:
            f.write(response.content)
        if verbose:
            logger.info(f'Downloaded {full_url} to {full_path}')
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to download file {full_url}: {e}")


def process_files(dir_url, i_pmt, n_files, i_hv, hv, cut, is_noisy_pmt, verbose):
    logger = configure_logger(i_pmt, hv)
    charge = pd.Series(dtype = float, index=range(n_files))
    charge[:] = np.nan

    local_base_dir = "data"
    remote_base_url = 'https://s3.cloud.infn.it/v1/AUTH_2ebf769785574195bde2ff418deac08a/cygno-analysis/GIN_PMT/'
    local_dir = os.path.join(local_base_dir, dir_url.replace(remote_base_url, '').split('Autosave')[0])
    
    download_tasks = []
    for i_file in range(n_files):
        file_number = f"{i_file:05d}"
        file_name = f"C1--n16deg--{file_number}.trc"
        full_url = f"{dir_url}{file_name}"
        full_path = os.path.join(local_dir, file_name)

        if not os.path.exists(full_path):
            download_tasks.append((full_url, full_path))

    if download_tasks:
            with ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [executor.submit(download_file, url, path, logger, verbose) for url, path in download_tasks]
                    for future in futures:
                        future.result()

    for i_file in range(n_files):
        file_number = f"{i_file:05d}"
        file_name = f"C1--n16deg--{file_number}.trc"
        full_path = os.path.join(local_dir, file_name)
        
        if verbose:
            logger.info(f'Analysing {file_name} from local file')

        try:
            with open(full_path, 'rb') as f:
                data_raw = lecroyparser.ScopeData(data=f.read(), sparse=-1)
        except FileNotFoundError as e:
            logger.warning(f"Failed to open local file {full_path}: {e}")
            continue
            
        data_raw_rs = scipy_signal.resample(data_raw.y, USAMPLE, data_raw.x)
        data = pd.DataFrame({'Time (s)': data_raw_rs[1], 'Amplitude (V)': data_raw_rs[0]})

        min_index = data['Amplitude (V)'].iloc[20:-20].idxmin()
        if pd.isna(min_index) or min_index <= 11 or data['Amplitude (V)'].iloc[min_index] >= cut:
            logger.warning(f"Warning: minimum not found for {PMT_NAMES[i_pmt-1]}, {hv} V, file {file_name}")
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
                    logger.warning(f"Warning: Parameter {param_name} not between bounds for {PMT_NAMES[i_pmt-1]}, {hv} V, file {file_name}")
            continue

        start_index = c_index - (20 if not is_noisy_pmt else 13)
        end_index = (data['Amplitude (V)'].iloc[min_index:] >= 0.04 * a / 2.5).idxmax() + (30 if not is_noisy_pmt else 20)
        if start_index <= 0 or end_index >= data.shape[0] :
            logger.warning(f"Warning: integration range out of data range, skipping {PMT_NAMES[i_pmt-1]}, {hv} V, file {file_name}")
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
    logger.handlers.clear()
    return (i_hv, hv, mean_charge, std_dev_charge)

def process_pmt(i_pmt, pmt_cuts, verbose):
    file_path = f'./data_info/data_directories_pmt_{i_pmt}.txt'
    dir_url_list, hv_list, n_files_list = read_parameters(file_path)
    print(f"Processing PMT {i_pmt}. Please check log file in outputs/logs/ for status updates.")
    charges = pd.DataFrame(columns=['mean_charge', 'std_dev_charge', 'hv'])
    charges[:] = np.nan

    is_noisy_pmt = PMT_NAMES[i_pmt - 1] in ["PMT_2", "PMT_4"]
    cuts = pmt_cuts[i_pmt - 1]

    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_files, dir_url, i_pmt, n_files, i_hv, hv, cuts[i_hv], is_noisy_pmt, verbose)
                       for i_hv, (hv, dir_url, n_files) in enumerate(zip(hv_list, dir_url_list, n_files_list))]

            for future in as_completed(futures):
                try:
                    i_hv, hv, mean_charge, std_dev_charge = future.result()
                    charges.loc[i_hv] = [mean_charge, std_dev_charge, float(hv)]
                except Exception as e:
                    logging.error(f"An error occurred: {e}")

        output_file = f'outputs/lists/pmt_{i_pmt}_data_list.pkl'

        with open(output_file, 'wb') as file:
            pickle.dump(charges, file)

        print(f"PMT {i_pmt} processed successfully.")

    finally:
        logging.shutdown()

def plot_charge_curves(pmt_data_list, pmt_names):
    plt.figure(figsize=(12, 9))
    custom_palette = sns.color_palette("Set1", n_colors=len(pmt_data_list))

    for i, charge_data in enumerate(pmt_data_list):
        hv_values = charge_data['hv']
        mean_charge_values = abs(charge_data['mean_charge'])
        std_dev_charge_values = charge_data['std_dev_charge']
        label = f'{pmt_names[i]}'  
        color = custom_palette[i]

        sns.lineplot(x=hv_values, y=mean_charge_values, label=label, color=color)
        plt.errorbar(x=hv_values, y=mean_charge_values, yerr=std_dev_charge_values, linestyle='',
                     marker='o', markersize=5, color=color)

    plt.xlabel('HV (V)')
    plt.ylabel('Mean Charge (pC)')
    plt.title('GIN PMT gain curves')
    plt.xticks(rotation=45)
    plt.yscale("log")
    plt.legend(title='PMT')
    plt.savefig('outputs/gain_curves.png', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Process PMT data.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    list_directories()
    pmt_cuts = load_pmt_cuts()
    
    for i_pmt in range(1,5):
            process_pmt(i_pmt, pmt_cuts, args.verbose)

    pmt_data_list, pmt_names = load_pickle_lists()
    plot_charge_curves(pmt_data_list, pmt_names)

    print("All tasks completed.")

if __name__ == "__main__":
    main()