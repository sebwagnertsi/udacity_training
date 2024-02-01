'''
This module contains functions to detect churn of customers.
The functions include the loading of data, as well as the exploratory data analysis,
feature calculation and model training

@Author Sebastian Wagner
@Date Jan 2024
'''
from typing import List
import os
import logging

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns
from config import Configuration

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

logging.basicConfig(
    filename=f'./{Configuration.log_folder}/churn.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s: %(message)s'
)


def import_data(path: str):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        print(f"No file could be found at: {path}")
        logging.error("No file could be found at: %s", path)
    return None


def construct_target(data: pd.DataFrame):
    '''
    Creates a column for the target (=y) values
    '''
    data[Configuration.target_attribute_name] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)


def perform_eda(data: pd.DataFrame):
    '''
    perform eda on data and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    try:
        plt.figure(figsize=(20, 10))
        data[Configuration.target_attribute_name].hist()
        plt.savefig(
            f"{Configuration.eda_target_folder}/{Configuration.target_attribute_name}.png")

        plt.figure(figsize=(20, 10))
        data['Customer_Age'].hist()
        plt.savefig(f"{Configuration.eda_target_folder}/customer_age.png")

        plt.figure(figsize=(20, 10))
        data.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(f"{Configuration.eda_target_folder}/marital_status.png")

        plt.figure(figsize=(20, 10))
        sns.histplot(data['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(f"{Configuration.eda_target_folder}/Total_Trans_ct.png")

        plt.figure(figsize=(20, 10))
        sns.heatmap(data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig(f"{Configuration.eda_target_folder}/heatmap.png")

        logging.info("Plots have been saved")
    except FileNotFoundError as fnf:
        print("A file could not be written: %s", fnf)
    except Exception as ex:
        logging.error("A Problem has occurred while saving plots: %s", ex)


def encoder_helper(
        data: pd.DataFrame,
        category_lst: List[str],
        target_attrib: str):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for all categorical attributes
    '''

    def add_categorized(data: pd.DataFrame, cat: str, target: str):
        '''
        Calculates the mean of a categorical attribute with respect to the target attribute.

        Input:
        df: dataframe
        cat: string name of the categorical attribute
        target_attrib: string name of the target attribute

        Output:
        Adds an attribute that contains the mean category value for each row
        '''
        groups = data.groupby(cat).mean()[target]

        def get_groupval(row):
            '''
            Returns the group result for a single row.
            Input:
            row: a dataframe row
            Return:
            the categorical average for that row
            '''
            return groups.loc[row[cat]]

        data[f"{cat}_{target}"] = data.apply(get_groupval, axis=1)

    for cat in category_lst:
        add_categorized(data, cat, target=target_attrib)


def perform_feature_engineering(data: pd.DataFrame):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    y = data[Configuration.target_attribute_name]

    X = pd.DataFrame()

    try:
        categories = ['Gender',
                      'Education_Level',
                      'Marital_Status',
                      'Income_Category',
                      'Card_Category'
                      ]
        encoder_helper(data, category_lst=categories,
                       target_attrib=Configuration.target_attribute_name
                       )
        logging.info(
            "Categorical mean value columns have been added for: %s",
            categories)
    except KeyError as key:
        logging.error(
            "Category_lst contains column names that do not exist: %s",
            key)

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    try:
        X[keep_cols] = data[keep_cols]
        logging.info("Dataframe has been reduced to columns: %s", keep_cols)
    except KeyError as key:
        logging.error(
            "keep_cols contains columns that do not exist: %s",
            key)

    return X, y


def classification_report_image(X_test, y_train,
                                y_test,
                                y_train_predictions_lr,
                                y_train_predictions_rf,
                                y_test_predictions_lr,
                                y_test_predictions_rf,
                                cv_randomforest,
                                logisticreg):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_predictions_rf))
    print('train results')
    print(classification_report(y_train, y_train_predictions_rf))
    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_predictions_lr))
    print('train results')
    print(classification_report(y_train, y_train_predictions_lr))

    lrc_plot = plot_roc_curve(logisticreg, X_test, y_test)

    try:
        plt.figure(figsize=(15, 8))
        axis = plt.gca()
        plot_roc_curve(
            cv_randomforest.best_estimator_,
            X_test,
            y_test,
            ax=axis,
            alpha=0.8)
        lrc_plot.plot(ax=axis, alpha=0.8)
        plt.savefig(f"{Configuration.report_folder}/Roc_curve.png")
    except FileNotFoundError as fnf:
        logging.error("The file could not be written: %s", fnf)


def feature_importance_plot(model, X_data):
    '''
    creates and stores the feature importances in the configured results folder
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    try:
        plt.savefig(f"{Configuration.report_folder}/feature_importance.png")
    except FileNotFoundError as fnf:
        logging.error("The file could not be written: %s", fnf)


def train_models(X_train, X_test, y_train):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    randomforest = RandomForestClassifier(random_state=42)
    logisticreg = LogisticRegression(solver='lbfgs',
                                     max_iter=Configuration.lr_max_iterations)

    cv_randomforest = GridSearchCV(
        estimator=randomforest,
        param_grid=Configuration.param_grid,
        cv=5)
    cv_randomforest.fit(X_train, y_train)
    logisticreg.fit(X_train, y_train)

    y_train_predictions_rf = cv_randomforest.best_estimator_.predict(X_train)
    y_test_predictions_rf = cv_randomforest.best_estimator_.predict(X_test)
    y_train_predictions_lr = logisticreg.predict(X_train)
    y_test_predictions_lr = logisticreg.predict(X_test)

    return y_train_predictions_rf, y_test_predictions_rf, y_train_predictions_lr, \
        y_test_predictions_lr, cv_randomforest, logisticreg


def store_model(model, filename):
    '''
    Stores a model at a certain file name in the designated model folder

    Input:
    model - the model to store
    filename - the filename to store it under
    '''
    joblib.dump(model, f"{Configuration.model_folder}/{filename}")


if __name__ == "__main__":

    # Step 0: Load the data from the csv file into dataframe
    churn_df = import_data(Configuration.data_source_file)
    construct_target(churn_df)

    # Step 1: EDA
    perform_eda(churn_df)

    # Step 2: Feature engineering
    X, y = perform_feature_engineering(churn_df)

    # Step 3: Model training and Prediction
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Step 4: Prediction
    y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, \
        y_test_preds_lr, cv_rfc, lrc = train_models(
            X_train, X_test, y_train)

    # Step 5: Evaluation
    classification_report_image(X_test, y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                cv_rfc, lrc)
    feature_importance_plot(cv_rfc, X_train)

    # Step 6: Save the best models to disk
    store_model(cv_rfc.best_estimator_, "rfc_model.pkl")
    store_model(lrc, "logistic_model.pkl")
