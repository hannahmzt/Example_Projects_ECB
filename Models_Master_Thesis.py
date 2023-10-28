import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso, LinearRegression
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from Performance import QLIKE

#######################################################################
# OLS
def OLS(y_train, X_train, y_test, X_test, y_vali, X_vali):
    '''
    input are the test, train and validation sets 
    calculates OLS 
    output: mse, qlike, predictions on test set
    '''
    # training data is here training and validation set
    # because no need to tune hyperparameters
    y_t = np.concatenate((y_train, y_vali), axis = 0)
    X_t = np.concatenate((X_train, X_vali), axis = 0)

    #introduce OLS model
    #model_ols = sm.OLS(y_t, X_t)
    model_ols = LinearRegression()

    # train model
    ols_train = model_ols.fit(X_t, y_t)

    # test model
    ols_pred = ols_train.predict(X_test)

    # mse
    mse = mean_squared_error(y_test, ols_pred)
    
    # qlike
    qlike = QLIKE(y_test, ols_pred)

    return mse, qlike, ols_pred

#######################################################################
# LASSO
def LASSO(y_train, X_train, y_test, X_test, y_vali, X_vali):
    '''
    input: test, train and validation sets
    calculates OLS with lasso, hyperparameter training with gridsearch using validation set
    output: mse, qlike, predictions on test set
    '''
    # alpha is lambda and needs to be tuned
    lasso = Lasso(max_iter = 1000)

    # Range of alpha values for grid search
    # IMPORTANT: check selection of alphas
    param_grid = {'alpha': [0.001, 0.002, 0.01, 0.1, 1, 10]}

    # Create GridSearchCV object
    grid_search = GridSearchCV(lasso, param_grid, cv = 5, scoring = 'neg_mean_squared_error',
                             n_jobs = -1)

    # fit to validation data
    grid_search.fit(X_vali, y_vali)

    # best alpha
    best_alpha = grid_search.best_params_['alpha']

    # Train LASSO
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X_train, y_train)

    # lasso with testset
    lasso_pred = lasso.predict(X_test)

    # mse
    mse = mean_squared_error(y_test, lasso_pred)
    
    # qlike
    qlike = QLIKE(y_test, lasso_pred)

    return mse, qlike, lasso_pred

#######################################################################
# XGBoost

def XGB(y_train, X_train, y_test, X_test, y_vali, X_vali):
    '''
    input: test, train and validation sets
    calculates XGBoost, hyperparameter training with gridsearch using validation set
    output: mse, qlike, predictions on test set
    '''
    # Grid search for meta-parameters
    xgb_param_grid = {
      'n_estimators': [100, 200],  # ensemble size or number of gradient steps
      'max_depth': [5, 10],   # max depth of decision trees
      'learning_rate': [0.1, 0.01]}  # learning rate


    model = xgb.XGBRegressor(random_state=42)

    # grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=xgb.XGBRegressor(), param_grid=xgb_param_grid,
                        scoring='neg_mean_squared_error', cv = 3, n_jobs=-1)

    # fit to training data
    grid_search.fit(X_vali, y_vali)

    # Get best Hyperparameters
    best_params = grid_search.best_params_

    # train model
    xgb_model = xgb.XGBRegressor(**best_params, random_state=42)
    xgb_model.fit(X_train, y_train)

    # test model
    xgb_pred = xgb_model.predict(X_test)

    # MSE
    mse = mean_squared_error(y_test, xgb_pred)
    
    # qlike
    qlike = QLIKE(y_test, xgb_pred)

    return mse, qlike, xgb_pred


#######################################################################
# Neuronal Network
#######################################################################
# Multilayer Perceptron

# initalize MLP Model
class MLPModel(nn.Module):
    def __init__(self, input_size , hidden_size, output_size):
        super(MLPModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        # Define hidden layers
        # nn.Linear automatically include both weight and bias terms
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        
        # activation function, here: ReLU
        # ReLU Rectified Linear Unit max(0,x)
        # Zhang used the same
        self.relu = nn.ReLU()
        
        # Define output layer
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # data flows forward through layers
        # bias terms are applied to each neuron in the hidden layers
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# initalize MPL Estimator for GridSearch
class MLPEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_size=64, num_epochs=50, lr = 0.001, input_size = 21, output_size = 1, loader=None):
        # hyperparameters are hidden_size, num_epochs and learning rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.loader = loader

    def fit(self, X, y):
        model = MLPModel(self.input_size, self.hidden_size, self.output_size)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.num_epochs):
            for batch_x, batch_y in self.loader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model parameters

        self.model = model

    def predict(self, X):
        # calculate predictions
        with torch.no_grad():
            return self.model(X).numpy()

# function for MLP
def MLP(y_train, X_train, y_test, X_test, y_vali, X_vali):
    '''
    input: test, train and validation sets
    calculates MLP, hyperparameter training with gridsearch using validation set
    output: mse, qlike, predictions on test set
    '''
    # convert data to PyTorch tensors
    # dimension needs to be (n_samples, features)
    X_train_new = torch.tensor(X_train, dtype=torch.float32)
    y_train_new = torch.tensor(y_train, dtype=torch.float32)
    X_test_new = torch.tensor(X_test, dtype=torch.float32)
    X_vali_new = torch.tensor(X_vali, dtype=torch.float32)
    y_vali_new = torch.tensor(y_vali, dtype=torch.float32)

    # Create DataLoader objects for batching
    batch_size = 64
    dataset = TensorDataset(X_train_new, y_train_new) # first input_features and then labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True)

    input_size = 21  # Input dimension (number of features)
    output_size = 1  # Output dimension (1 for regression)

    # Grid Search for hyperparameter tuning
    param_grid = {
    'hidden_size': [32, 64, 128],  # Define hidden size options
    'num_epochs': [20, 50, 75],  # Set a fixed value for num_epochs
    'lr': [0.001, 0.002]
    }

    grid_search = GridSearchCV(
        estimator=MLPEstimator(input_size = input_size, output_size = output_size, loader = loader),
        param_grid=param_grid,
        cv=3,  # Use 3-fold cross-validation
        scoring='neg_mean_squared_error',  # Use MSE as the evaluation metric
        n_jobs=-1  # Use all available CPU cores for parallel processing
    )

    # Fit the grid search to the data
    grid_search.fit(X_vali_new, y_vali_new)

  
    hidden_size = grid_search.best_params_['hidden_size'] # Number of LSTM units

    model = MLPModel(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), grid_search.best_params_['lr'])  # explain Adam optimizer in thesis

    num_epochs = grid_search.best_params_['num_epochs']
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_x)  
            loss = criterion(outputs, batch_y)  
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

    # make prediction
    predictions = model(X_test_new) 

    # calculate MSE
    mse = mean_squared_error(y_test, predictions.detach().numpy())
    
    # qlike
    qlike = QLIKE(y_test, predictions.detach().numpy())

    return mse, qlike, predictions.detach().numpy()


#######################################################################
# LSTM

# create LSTM Model with one layer
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        
        #Define LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        #becuase batch_first = True input shape is (batch, sequence, features)
        
        # Define output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # initalize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x)
        
        out = self.fc(out[:, -1, :])  # Get the output from the last time step
        return out


# create LSTM Estimator for Hyperparameter tuning
# for using GridSearch "fit" and 'predict' is needed

class LSTMEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_size=64, num_epochs=50, lr = 0.001, input_size =1, output_size=1, loader = None):
        # hyperparameters are hidden_size, num_epochs and learning rate
        self.hidden_size = hidden_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.input_size = input_size
        self.output_size = output_size
        self.loader = loader

    def fit(self, X, y):
        model = LSTMModel(self.input_size, self.hidden_size, self.output_size)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        for epoch in range(self.num_epochs):
            for batch_x, batch_y in self.loader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(batch_x)
                #outputs = model(batch_x.unsqueeze(-1))  # Add a channel dimension
                #loss = criterion(outputs, batch_y.unsqueeze(-1))  # Add a channel dimension
                loss = criterion(outputs, batch_y)
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model parameters

        self.model = model

    def predict(self, X):
        with torch.no_grad():
            return self.model(X).numpy()

# function for LSTM
def LSTM(y_train, X_train, y_test, X_test, y_vali, X_vali):
    '''
    input: test, train and validation sets
    calculates LSTM, hyperparameter training with gridsearch using validation set
    output: mse, qlike, predictions on test set
    '''
    
    # convert data to PyTorch tensors
    # dimension needs to be (batch, sequence, feature)
    X_train_new = torch.tensor(X_train, dtype=torch.float32)
    X_train_new = torch.reshape(X_train_new, (X_train_new.shape[0], X_train_new.shape[1], 1))
    y_train_new = torch.tensor(y_train, dtype=torch.float32)
    X_test_new = torch.tensor(X_test, dtype=torch.float32)
    X_test_new = torch.reshape(X_test_new, (X_test_new.shape[0], X_test_new.shape[1], 1))
    X_vali_new = torch.tensor(X_vali, dtype=torch.float32)
    X_vali_new = torch.reshape(X_vali_new, (X_vali_new.shape[0], X_vali_new.shape[1], 1))
    y_vali_new = torch.tensor(y_vali, dtype=torch.float32)

    # Create DataLoader objects for batching
    batch_size = 32
    dataset = TensorDataset(X_train_new, y_train_new) # first input_features and then labels
    loader = DataLoader(dataset, batch_size=batch_size, shuffle = True)

    input_size = 1  # Input dimension (1 for univariate time series)
    output_size = 1  # Output dimension (1 for regression)

    # Grid Search for hyperparameter tuning
    param_grid = {
    'hidden_size': [32, 64, 128],  # Define hidden size options
    'num_epochs': [20, 50, 75],  # Set a fixed value for num_epochs
    'lr': [0.001, 0.002]
    }

    grid_search = GridSearchCV(
    estimator=LSTMEstimator(loader=loader),
    param_grid=param_grid,
    cv=3,  # Use 3-fold cross-validation
    scoring='neg_mean_squared_error',  # Use MSE as the evaluation metric
    n_jobs=-1  # Use all available CPU cores for parallel processing
    )

    # Fit the grid search to the data
    grid_search.fit(X_vali_new, y_vali_new)


    hidden_size = grid_search.best_params_['hidden_size'] # Number of LSTM units

    model = LSTMModel(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), grid_search.best_params_['lr'])  # explain Adam optimizer in thesis

    num_epochs = grid_search.best_params_['num_epochs']
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_x)  
            loss = criterion(outputs, batch_y)  
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

    # make prediction
    predictions = model(X_test_new) 

    # calculate MSE
    mse = mean_squared_error(y_test, predictions.detach().numpy())
    
    # qlike
    qlike = QLIKE(y_test, predictions.detach().numpy())
    

    return mse, qlike, predictions.detach().numpy()