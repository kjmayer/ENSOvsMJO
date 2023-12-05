#!/bin/bash

# Change these values as needed
AVG_START=2
AVG_END=6
LEAD_START=22
LEAD_END=31

# Load necessary modules
module purge
source /glade/work/wchapman/miniconda3/bin/activate
conda activate tf2-env

# Loop over LEAD values
for ((lead=$LEAD_START; lead<$LEAD_END; lead++)); do
    # Loop over AVG_X values
    for ((avg=$AVG_START; avg<=$AVG_END; avg++)); do
        EXP_NAME="DOY_LEAD_${lead}_AVG_${avg}_"
        
        # Loop over seed values from 2 to 5
        for seed in {1..5}; do
            python trainANN_gordon.py --LEAD $lead --EXP_NAME $EXP_NAME --Y_RUNMEAN $avg --SEED $seed --CUSTOM_RUN
        done
    done
done
