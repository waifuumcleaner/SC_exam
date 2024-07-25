#!/usr/bin/env python
# coding: utf-8

import os

def list_directories():
    data_dir = "./data/"
    outputs_main_dir = "./outputs/"
    lists_dir = os.path.join(outputs_main_dir, "condor_lists")
    outputs_dir = os.path.join(outputs_main_dir, "condor_outputs")
    logs_dir = os.path.join(outputs_main_dir, "condor_logs")
    errors_dir = os.path.join(outputs_main_dir, "condor_errors")

    for dir_path in [data_dir, lists_dir, outputs_dir, logs_dir, errors_dir]:
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
            
if __name__ == "__main__":
    list_directories()