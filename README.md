# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The goal of this project is to identify bank customers that are likely to switch to another bank.
Given is a historical dataset of previous and current customers together with a set of attributes in csv format.
The program runs through five steps:
- Step 1: EDA
- Step 2: Feature engineering
- Step 3: Model training
- Step 4: Prediction
- Step 5: Evaluation
- Step 6: Save the resulting models


## Files and data description
- data/bank_data.csv contains the customer information in csv format
- churn_library.py contains all necessary functionality to execute the process.
- churn_library_test.py contains the necessary pytest tests for the library

## Running Files
You can run the whole process by executing
~~~python churn_library.py~~~

You can run the tests by running pytest from the root folder and
pipe the results into a log file:
~~~pytest tests/churn_library_test.py > test.log~~~

If you want to see the print statements add the -s flag:
~~~pytest tests/churn_library_test.py -s~~~
