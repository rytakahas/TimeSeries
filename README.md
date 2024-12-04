### Objective: Time Series Regression for Population

This project implements time series regression techniques for forecasting <br>
population using three different models: **SARIMAX**, **XGBoost**, and **LSTM** <br>

The workflow includes data preprocessing, model training, hyperparameter optimization, <br>
 evaluation, and visualization of the results.

### Steps
1: Data Exploration and Preprocessing <br>
2: Model Training with SARIMAX, XGBoost, and LSTM <br>
3: Evaluation Metrics and Visualization <br>
4: Deliverables <br>
<br>
<br>

1. **Data Preparation**
    - Load the population dataset.
    - Analyze seasonality and stationarity of the time series data.
    - Split the dataset into training and testing sets.

2. **Model Training**
    - Split the dataset into training and testing sets.
    - Standard Scaling ($\mu$ = 0, $\sigma$ = 1) for XGBoost and LSTM
    - Trained with SARIMAX, XGBoost (optuna), and LSTM 

3. **Evaluation**
    - Evaluate MSE (Normalized MSE), MAE (Normalized MAE), and R2
    - Visualized the prediction results

4. **Deliverables**
    - SARIMAX, and XGBoost trained models were saved to 'pkl' files, <br> 
    and LSTM to 'keras' for reproducing test results.