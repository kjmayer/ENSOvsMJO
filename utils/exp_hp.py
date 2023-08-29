def get_hp(experiment_name):
    
    experiments = {
        
        ## Experiment 1:
        'default': {
            'INPUT': ['TS_SST_ONI','RMM1_CESM2','RMM2_CESM2'],
            'OUTPUT': ['TS_Z500a'],
            'LEAD': 7,
            'X_ADDITIONAL_DAYS': 15,
            'Y_RUNMEAN': 7,
            'HIDDENS1': [8], 
            'HIDDENS2': [8],
            'RIDGE1': 0,
            'RIDGE2': 0,
            'BATCH_SIZE': 32,
            'LR': 0.001,
            'DROPOUT_RATE': 0,
            'PATIENCE': 20,
            'GLOBAL_SEED': 99,
        },
        
        ## Experiment 1:
        'exp1': {
            'INPUT': ['TS_SST_ONI','RMM1_CESM2','RMM2_CESM2'],
            'OUTPUT': ['TS_Z500a'],
            'LEAD': 7,
            'X_ADDITIONAL_DAYS': 15,
            'Y_RUNMEAN': 7,
            'HIDDENS1': [8], 
            'HIDDENS2': [8],
            'RIDGE1': 0,
            'RIDGE2': 0,
            'BATCH_SIZE': 32,
            'LR': 0.001,
            'DROPOUT_RATE': 0,
            'PATIENCE': 20,
            'GLOBAL_SEED': 99,
        },
        
        ## Experiment 1:
        'exp2': {
            'INPUT': ['TS_SST_ONI','RMM1_CESM2','RMM2_CESM2'],
            'OUTPUT': ['TS_Z500a'],
            'LEAD': 14,
            'X_ADDITIONAL_DAYS': 15,
            'Y_RUNMEAN': 7,
            'HIDDENS1': [8], 
            'HIDDENS2': [8],
            'RIDGE1': 0,
            'RIDGE2': 0,
            'BATCH_SIZE': 32,
            'LR': 0.001,
            'DROPOUT_RATE': 0,
            'PATIENCE': 20,
            'GLOBAL_SEED': 99,
        },
        
    }

    return experiments[experiment_name]
