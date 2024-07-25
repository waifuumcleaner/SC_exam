#!/bin/bash

for pmt in {1..4}; do
    cat <<EOF > analysis_pmt_${pmt}.sub
universe   = vanilla
executable = exec_condor.sh
arguments  = \$(dir_url) \$(i_pmt) \$(hv) \$(n_files)
output     = ../condor_outputs/pmt_\$(i_pmt)_\$(hv)_analysis.out
log        = pmt_\$(i_pmt)_\$(hv)_analysis.log
error      = ../condor_errors/pmt_\$(i_pmt)_\$(hv)_analysis.error

should_transfer_files = YES
when_to_transfer_output = ON_EXIT

transfer_input_files = ../../analysis_condor.py, ../../data_info/pmt_cuts.pkl, ../../data_info/voltages_list.pkl
transfer_output_files = pmt_data_list.pkl
initial_dir = outputs/condor_logs
transfer_output_remaps = "pmt_data_list.pkl=../condor_lists/pmt_\$(i_pmt)_\$(hv)_data_list.pkl"
max_retries = 3

+OWNER = "condor"
queue dir_url, i_pmt, hv, n_files from data_info/data_directories_pmt_${pmt}.txt
EOF

    condor_submit -spool analysis_pmt_${pmt}.sub
done
