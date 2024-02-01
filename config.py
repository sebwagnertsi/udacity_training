'''
This singleton class provides the configuration for the project.

Author: Sebastian Wagner
Date: Feb 2024
'''
class Configuration:
    '''
    This class holds all parameters for the execution of the analysis steps.
    '''
    target_attribute_name = 'Churn'
    eda_target_folder = r'images/eda'
    log_folder = r'logs'
    model_folder = r'models'
    report_folder = r'images/results'
    data_source_file = r"./data/bank_data.csv"

    lr_max_iterations = 3000
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }