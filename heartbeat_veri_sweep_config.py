sweep_config = {
                'method': 'bayes',
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize' 
                },
                'early_terminate' : {
                    "type": 'hyperband',
                    "s": 2,
                    "eta": 3,
                    "max_iter": 27
                },
  
                'parameters': {
                    'epochs': {
                        'values': [1]
                    },
                    'batch_size': {
                        # 'values': [5]
                        'min' : 1,
                        'max' : 32
                    },
                    'momentum': {
                        'values': [0.9]
                    },
                    'weight_decay': {
                        'values': [1e-5]
                    },
                    'learning_rate': {
                        # 'values': [0.07112]
                        'min' : 1e-5,
                        'max' : 1e-1
                    },
                    'optimizer': {
                        'values': ['adam']
                    },
                    'scheduler': {
                        'values': ["Cosine Annealing"]
                    },
                    'criterion': {
                        'values': ['binary_cross_entropy']  
                    },
                    'use_SAM':{
                        'values' : [False]
                    }
                }
            }
             