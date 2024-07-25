import pickle
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import sys
sns.set_theme()

def load_pickle_lists():
    try:
        pmt_data_list = []
        pmt_names = [f'PMT_{i}' for i in range(1,5)]

        for i_pmt in range(1, 5):
            combined_data = []

            # Find all .pkl files for the current PMT
            file_pattern = f'outputs/condor_lists/pmt_{i_pmt}_*_data_list.pkl'
            pkl_files = glob.glob(file_pattern)

            # Load and combine data from all found files
            for pkl_file in pkl_files:
                with open(pkl_file, 'rb') as file:
                    data = pickle.load(file)
                    combined_data.append(data)
                    print(f'{pkl_file} successfully loaded')

            # Combine all data into a single DataFrame
            if combined_data:
                combined_df = pd.concat(combined_data, ignore_index=True)
                pmt_data_list.append(combined_df)
            else:
                print(f'No data files found for PMT_{i_pmt}')

        return pmt_data_list, pmt_names

    except Exception as e:
        print(f"Error loading pickle file: {e}")
        sys.exit(1)
        
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
    
if __name__ == "__main__":
    pmt_data_list, pmt_names = load_pickle_lists()
    plot_charge_curves(pmt_data_list, pmt_names)
