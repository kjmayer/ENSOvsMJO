#!/bin/bash

# Change these values as needed
AVG_START=7
AVG_END=31
LEAD_START=7
LEAD_END=31

# Load necessary modules
module purge
source /glade/work/wchapman/miniconda3/bin/activate
conda activate tf2-env

# Loop over LEAD values
for ((lead=$LEAD_START; lead<$LEAD_END; lead++)); do
    # Loop over AVG_X values
    for ((avg=$AVG_START; avg<=$AVG_END; avg++)); do
        EXP_NAME="LEAD_${lead}_AVG_${avg}_"
        python trainANN_gordon.py --LEAD $lead --EXP_NAME $EXP_NAME --Y_RUNMEAN $avg --CUSTOM_RUN
    done
done
