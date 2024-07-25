#!/bin/bash
dir_url=$1
i_pmt=$2
hv=$3
n_files=$4

if [[ ! -f analysis_condor.py || ! -f pmt_cuts.pkl || ! -f voltages_list.pkl ]]; then
    echo "Required files are missing!"
    exit 1
fi

python3 analysis_condor.py ${dir_url} ${i_pmt} ${hv} ${n_files}
if [ $? -ne 0 ]; then
    echo "analysis_condor.py failed to execute."
    exit 1
fi

echo "Current directory: $(pwd)"
ls -l
