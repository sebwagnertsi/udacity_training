'''
This test file provides pytest for all the functions in churn_library.

Author: Sebastian Wagner
Date: Feb 2024
'''
import logging
import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import churn_library as cls

from config import Configuration

logging.basicConfig(
    filename='churn_tests.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s: %(message)s'
)


@pytest.fixture
def data_path():
    '''
    Provides the path for the csv file.
    '''
    return "./data/bank_data.csv"


@pytest.fixture
def dataframe(data_path):
    '''
    Provides the imported csv data for the tests.
    '''
    try:
        data = cls.import_data(data_path)
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err
    return data[0:200]


@pytest.fixture
def testdata():
    '''
    Provides a smaller test dataframe for actual unit testing.
    '''
    data = {
        'Category1': ['A', 'B', 'C', 'A', 'A', 'B', 'C', 'A'],
        'Category2': ['X', 'X', 'Y', 'Z', 'X', 'Y', 'X', 'Z'],
        'Numeric1': np.random.rand(8),
        'Numeric2': np.random.rand(8),
        'Churn': [True, False, True, False, True, False, True, False]
    }
    return pd.DataFrame(data)


def test_import(data_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data = cls.import_data(data_path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_construct_target(dataframe):

    cls.construct_target(dataframe)
    assert 'Churn' in dataframe.columns


def test_eda(dataframe):
    '''
    test perform eda function
    '''
    try:
        cls.construct_target(dataframe)
        cls.perform_eda(dataframe)
        logging.info("Testing EDA functionality: SUCCESS")
    except Exception as err:
        logging.error("Testing EDA functionality FAILED:\n", err)
        raise err


def test_encoder_helper(testdata: pd.DataFrame):
    '''
    test encoder helper

    Encode the categorical data of the testdata fixture and check
    for the correct outcome.
    '''

    try:
        cls.encoder_helper(testdata,
                           category_lst=[
                               'Category1',
                               'Category2'
                           ],
                           target_attrib='Churn'
                           )
        logging.info("Testing encoder_helper functionality: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper functionality FAILED:\n", err)
        raise err

    assert testdata.shape[1] == 7
    assert testdata.shape[0] == 8
    assert 'Category1_Churn' in testdata.columns
    assert 'Category2_Churn' in testdata.columns


def test_perform_feature_engineering(dataframe: pd.DataFrame):
    '''
    test perform_feature_engineering
    '''

    # Previous steps:
    cls.construct_target(dataframe)

    # Step 2: Feature engineering
    try:
        X, y = cls.perform_feature_engineering(dataframe)
        logging.info(
            "Testing perform_feature_engineering functionality: SUCCESS")
    except Exception as err:
        logging.error("Testing encoder_helper functionality FAILED:\n%s", err)
        raise err
    assert 'Marital_Status_Churn' in X.columns
    assert 'Card_Category_Churn' in X.columns
    assert 'Attrition_Flag' not in X.columns


def test_train_models(dataframe):
    '''
    test train_models
    '''

    # Previous steps:
    cls.construct_target(dataframe)
    X, y = cls.perform_feature_engineering(dataframe)

    # Step 3: Model training and Prediction
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Set the configuration to something with less iterations to run faster
    Configuration.lr_max_iterations = 1000
    Configuration.param_grid = {
        'n_estimators': [10, 20],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

# Step 4: Prediction
    try:
        y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, \
            y_test_preds_lr, cv_rfc, lrc = cls.train_models(
                X_train, X_test, y_train)
        logging.info("Testing train_models functionality: SUCCESS")
    except Exception as err:
        logging.error("Testing train_models functionality FAILED:\n %s", err)
        raise err


def test_model_saving(dataframe):
    '''
    Runs the model saving part and logs potential errors.
    '''
    # Previous steps:
    cls.construct_target(dataframe)
    X, y = cls.perform_feature_engineering(dataframe)

    # Step 3: Model training and Prediction
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Set the configuration to something with less iterations to run faster
    Configuration.lr_max_iterations = 1000
    Configuration.param_grid = {
        'n_estimators': [10, 20],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, \
        y_test_preds_lr, cv_rfc, lrc = cls.train_models(
            X_train, X_test, y_train)

    try:
        cls.store_model(cv_rfc.best_estimator_, f"rfc_model.pkl")
        cls.store_model(lrc, f"logistic_model.pkl")
        logging.info("store_model: SUCCESS")
    except Exception as err:
        logging.error("Testing store_model functionality FAILED:\n%s", err)
        raise err
