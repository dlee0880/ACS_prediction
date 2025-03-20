import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from utils import plot_confusion_matrix, plot_roc, normalize
import keras_tuner as kt
from tensorflow import keras

def build_model(hp, param_grids, input_shape):
    num_dense_layers = hp.Int ('num_dense_layers', min_value=param_grids['num_dense_layers_min'], max_value=param_grids['num_dense_layers_max'])
    dense_dims = [hp.Choice('dense_dims', param_grids['dense_dims']) for i in range(num_dense_layers)]
    activation = hp.Choice('activation', param_grids['activation'])
    dropout_rate = hp.Float('dropout_rate', min_value=param_grids['dropout_rate_min'], max_value=param_grids['dropout_rate_max'], step=0.1)
    dnn_x = input_shape
    
    for i in range(num_dense_layers):
        dnn_x = Dense(dense_dims[i], activation=activation)(dnn_x)
        dnn_x = BatchNormalization()(dnn_x)
        dnn_x = Dropout(dropout_rate)(dnn_x)
    
    prob_outcome = Dense(1, activation='sigmoid')(dnn_x)
    model_dense = keras.Model(inputs=input_shape, outputs=prob_outcome)
    model_dense.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) 
    
    return model_dense

def random_search_dnn(param_grids, input_shape, x_train, y_train, x_test, y_test, column_name):
    """Random search for 100 trials
    Args:
        param_grids (dictionary): A dictionary to storage searching space for DNN architecture
    Returns:
        grid_search_params: A dictionary of searching space for next grid search step
    """
    tuner = kt.RandomSearch(lambda hp: build_model(hp, param_grids, input_shape), 
                            objective='val_accuracy', 
                            max_trials=50, 
                            directory='my_dir', 
                            project_name='random_search_project_' + column_name)
    
    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    
    tuner.search(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test), callbacks=[stop_early])
    model_dense = tuner.get_best_models()[0]
    model_dense.summary()
    grid_search_params_json = tuner.oracle.get_best_trials(num_trials=50)[0].hyperparameters.values
    
    grid_search_params = {
        'num_dense_layers_min': max(grid_search_params_json['num_dense_layers'] - 1, 1),
        'num_dense_layers_max': max(grid_search_params_json['num_dense_layers'] + 1, 0),
        'dense_dims': [grid_search_params_json['dense_dims']],
        'activation': ['relu', 'sigmoid', 'tanh'],
        'dropout_rate_min': max(grid_search_params_json['dropout_rate'] - 0.1, 0),
        'dropout_rate_max': max(grid_search_params_json['dropout_rate'] + 0.1, 0)
    }
    return grid_search_params

def grid_search_dnn(param_grids, input_shape, x_train, y_train, x_test, y_test, column_name):
    """Grid search for 27 trials
    Args:
        param_grids (dictionary): A dictionary to storage searching space for DNN architecture
    Returns:
        model_dense: Keras model storage learned weights
    """
    #Hyperparameter Optimization
    tuner = kt.GridSearch(lambda hp: build_model(hp, param_grids, input_shape), 
                          objective='val_accuracy', 
                          max_trials=27, 
                          directory='my_dir', 
                          project_name='grid_search_project_' + column_name)
    
    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    
    tuner.search(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test), callbacks=[stop_early])
    model_dense = tuner.get_best_models()[0]
    model_dense.summary()
    hps = tuner.oracle.get_best_trials(num_trials=27)[0].hyperparameters.values
    print(column_name + ' ---> HyperParameters: {}'.format(hps))
    return model_dense

def single_column_train_and_estimate_ps_score(model_name, column_name, profiles, parent_dir_dict):
    column_name_folder = column_name
    parent_dir = parent_dir_dict[model_name]

    if model_name in ["XGBOOST", "LGBM"]:
        """Train and estimate propensity score for single column
        Args:
            column_name (String): Name of current column for propensity score estimation
        Returns: 
            single_column_propensity_score_clipped_df(Data Frame): A data frame contains propensity score of current column
        """
        column_name_path = os.path.join(parent_dir, column_name_folder) 

        # Create the directory  to save model checkpoint
        try:
            os.mkdir(column_name_path) 
        except FileExistsError:
            print("Folder already exists")

        x, y = profiles.drop(columns = column_name), profiles[column_name]

        if model_name == "XGBOOST":
            data_dmatrix = xgb.DMatrix(data=x, label=y)
            data_dmatrix

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

            #Declaring parameters
            params = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'lambda': 10,
            'learning_rate': 1.0,
            'n_estimators': 100
            }

            # Parameters to optimize
            gridParams = {
            'max_depth': [2, 4, 6, 8],
            'lambda': [1, 2, 3, 4, 5, 6],
            'learning_rate': [0.2, 0.4, 0.6, 0.8, 1.0],
            'n_estimators' : [100, 150, 200, 250, 300]
            }

            # Create a classifier
            mdl = XGBRegressor(objective= 'binary:logistic',
                    learning_rate = params['learning_rate'],
                    n_estimators = params['n_estimators'], 
                    max_depth = params['max_depth'],
                    reg_lambda = params['lambda'],
                    seed=123)

            #create the grid
            grid = GridSearchCV(mdl, gridParams, verbose=1, cv=10, n_jobs=-1)

            #Search for optimal parameters
            grid.fit(x_train, y_train)

            params['max_depth'] = grid.best_params_['max_depth']
            params['lambda'] = grid.best_params_['lambda']
            params['learning_rate'] = grid.best_params_['learning_rate']
            params['n_estimators'] = grid.best_params_['n_estimators']

            print(f'Fitting with params: {params}')

            xgb_reg = XGBRegressor(**params)
            xgb_reg.fit(x_train, y_train)

            grid_file_path = column_name + '/grid.pkl'
            xgb_file_path = column_name + '/xgb.pkl'
            
            # Save grid
            joblib.dump(grid, grid_file_path)
            # Save lgb
            joblib.dump(xgb_reg, xgb_file_path)

            # Evaluate the model on the test set
            y_test_pred = xgb_reg.predict(x_test)
            y_test_pred_binary = np.round(y_test_pred).astype(int)
            accuracy_test_set = accuracy_score(y_test, y_test_pred_binary)
            print('TestSet Accuracy:', accuracy_test_set)
            plot_confusion_matrix(y_test, y_test_pred, ['Negative', 'Positive'])
            auc_test_set = plot_roc(y_test, y_test_pred)

            # Evaluate the model on the train set
            y_train_pred = xgb_reg.predict(x_train)
            y_train_pred_binary = np.round(y_train_pred).astype(int)
            accuracy_train_set = accuracy_score(y_train, y_train_pred_binary)
            print('TrainSet Accuracy:', accuracy_train_set)
            plot_confusion_matrix(y_train, y_train_pred, ['Negative', 'Positive'])
            auc_train_set = plot_roc(y_train, y_train_pred)

            # Evaluate the model on the whole dataset
            y_pred = xgb_reg.predict(x)
            y_pred_binary = np.round(y_pred).astype(int)
            accuracy_data_set = accuracy_score(y, y_pred_binary)
            print('DatatSet Accuracy:', accuracy_data_set)
            plot_confusion_matrix(y, y_pred, ['Negative', 'Positive'])
            auc_data_set = plot_roc(y, y_pred)

            single_column_propensity_score_df = pd.DataFrame(y_pred)
            
            # Histogram (for PS overlap)
            profiles['ps'] = y_pred
            sns.histplot(data=profiles, x='ps', hue=column_name)
            plt.show()         

        elif model_name == "LGBM":
            normalize(x, x.columns)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
            train_data = lgb.Dataset(x_train, label=y_train)

            # Model hyper-parameters (Default)
            params = {'max_depth' : -1,
                    'num_leaves': 64,
                    'learning_rate': 0.07,
                    'max_bin': 512,
                    'subsample_for_bin': 200,
                    'subsample': 1,
                    'subsample_freq': 1,
                    'colsample_bytree': 0.8,
                    'num_class' : 1,
                    'metric' : 'binary_error',
                    'verbosity' : 0
                    }

            # Parameters to optimize
            gridParams = {
                'learning_rate': [0.2, 0.4, 0.6, 0.8],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'num_leaves': [2, 4, 6, 8, 10, 20, 24, 28],
                'colsample_bytree' : [0.60, 0.65],
                #'reg_lambda' : [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
                }

            # Create a classifier
            mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
                    objective = 'binary',
                    n_jobs = 5, 
                    max_depth = params['max_depth'],
                    max_bin = params['max_bin'],
                    subsample_for_bin = params['subsample_for_bin'],
                    subsample = params['subsample'],
                    subsample_freq = params['subsample_freq'],
                    learning_rate = gridParams['learning_rate'],
                    n_estimators = gridParams['n_estimators'],
                    num_leaves = gridParams['num_leaves'],
                    colsample_bytree = gridParams['colsample_bytree'],
                    #reg_lambda = gridParams['reg_lambda']
                    )

            #Create the grid
            grid = GridSearchCV(mdl, gridParams, verbose=1, cv=10, n_jobs=-1)

            #Search for optimal parameters
            grid.fit(x_train, y_train)

            #Print the best parameters
            print(f'Best params: {grid.best_params_}')
            print(f'Best score: {grid.best_score_}')

            params['colsample_bytree'] = grid.best_params_['colsample_bytree']
            params['learning_rate'] = grid.best_params_['learning_rate']
            params['num_leaves'] = grid.best_params_['num_leaves']
            params['n_estimators'] = grid.best_params_['n_estimators']
            #params['reg_lambda'] = grid.best_params_['reg_lambda']

            print(f'Fitting with params: {params}')

            #Train the model on the selected parameters
            lgbm = lgb.train(params,
                            train_data,
                            num_boost_round=200,
                            verbose_eval= 4)

            grid_file_path = column_name_path  + '/grid.pkl'
            lgb_file_path = column_name_path  + '/lgb.pkl'
            
            # Save grid
            joblib.dump(grid, grid_file_path)
            # Save lgb
            joblib.dump(lgbm, lgb_file_path)

            # Evaluate the model on the test set
            y_test_pred = lgbm.predict(x_test)
            y_test_pred = np.clip(y_test_pred, 0, 1)
            y_test_pred_binary = np.round(y_test_pred).astype(int)
            accuracy_test_set = accuracy_score(y_test, y_test_pred_binary)
            print('TestSet Accuracy:', accuracy_test_set)

            # Plot on test set
            plot_confusion_matrix(y_test, y_test_pred, ['Negative', 'Positive'])
            auc_test_set = plot_roc(y_test, y_test_pred)

            # Evaluate the model on the train set
            y_train_pred = lgbm.predict(x_train)
            y_train_pred = np.clip(y_train_pred, 0, 1)
            y_train_pred_binary = np.round(y_train_pred).astype(int)
            accuracy_train_set = accuracy_score(y_train, y_train_pred_binary)
            print('TrainSet Accuracy:', accuracy_train_set)

            # Plot on train set
            plot_confusion_matrix(y_train, y_train_pred, ['Negative', 'Positive'])
            auc_train_set = plot_roc(y_train, y_train_pred)

            # Evaluate the model on the whole dataset
            y_pred = lgbm.predict(x)
            y_pred = np.clip(y_pred, 0, 1)
            y_pred_binary = np.round(y_pred).astype(int)
            accuracy_data_set = accuracy_score(y, y_pred_binary)
            print('DatatSet Accuracy:', accuracy_data_set)

            # Plot on whole dataset
            plot_confusion_matrix(y, y_pred, ['Negative', 'Positive'])
            auc_data_set = plot_roc(y, y_pred)

            single_column_propensity_score_df = pd.DataFrame(y_pred)
            
            # Histogram (for PS overlap)
            profiles['ps'] = y_pred
            sns.histplot(data=profiles, x='ps', hue=column_name)
            plt.show()  

    elif model_name in ["DNN", "LSTM"]:
        """Train and estimate propensity score for single column
        Args:
            column_name (String): Name of current column for propensity score estimation
        Returns: 
            single_column_propensity_score_df(Data Frame): A data frame contains propensity score of current column
        """
        x, y = profiles.drop(columns=[column_name]), profiles[column_name]
        if model_name == "DNN":
            normalize(x, x.columns)

            x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y, test_size=0.2, random_state=0)
            input_shape = keras.Input(shape=(x.shape[1],))
            
            #random search
            random_search_params = {
                'num_dense_layers_min': 1,
                'num_dense_layers_max': 6,
                'dense_dims': [8, 16, 32, 64, 128],
                'activation': ['relu', 'sigmoid', 'tanh'],
                'dropout_rate_min': 0.1,
                'dropout_rate_max': 0.9
            }
            grid_search_params = random_search_dnn(random_search_params, input_shape, x_train, y_train, x_test, y_test, column_name)
            
            print(column_name + " ---> grid_search_params:")
            print(grid_search_params)
                
            #grid search
            model_dense = grid_search_dnn(grid_search_params, input_shape, x_train, y_train, x_test, y_test, column_name)

            y_test_pred = model_dense.predict(x_test)
            y_test_pred_binary = np.round(y_test_pred).astype(int)
            accuracy_test_set = accuracy_score(y_test, y_test_pred_binary)
            print('TestSet Accuracy:', accuracy_test_set)
            plot_confusion_matrix(y_test, y_test_pred, ['Negative', 'Positive'])
            auc_test_set = plot_roc(y_test, y_test_pred)

            # Evaluate the model on the train set
            y_train_pred = model_dense.predict(x_train)
            y_train_pred_binary = np.round(y_train_pred).astype(int)
            accuracy_train_set = accuracy_score(y_train, y_train_pred_binary)
            print('TrainSet Accuracy:', accuracy_train_set)
            plot_confusion_matrix(y_train, y_train_pred, ['Negative', 'Positive'])
            auc_train_set = plot_roc(y_train, y_train_pred)

            # Evaluate the model on the whole dataset
            y_pred = model_dense.predict(x)
            y_pred_binary = np.round(y_pred).astype(int)
            accuracy_data_set = accuracy_score(y, y_pred_binary)
            print('DatatSet Accuracy:', accuracy_data_set)
            plot_confusion_matrix(y, y_pred, ['Negative', 'Positive'])
            auc_data_set = plot_roc(y, y_pred)

            single_column_propensity_score_df = pd.DataFrame(y_pred)
            
            # Histogram (for PS overlap)
            profiles['ps'] = y_pred
            sns.histplot(data=profiles, x='ps', hue=column_name)
            plt.show() 
        elif model_name == "LSTM":
            # Scaling data
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(x)
            x = pd.DataFrame(df_scaled)

            # Define the dimensions
            num_time_visits = 5
            num_features = x.shape[1]
            x_row_numbers = x.shape[0]
            actual_patient_nums = int(x_row_numbers/num_time_visits)

            # Reshape x(28305, 64) -> x(5661, 5, 64) and y(28305) -> y(5661, 5)
            x_reshape = x.values.reshape((actual_patient_nums, num_time_visits, num_features))
            y_reshape = y.values.reshape((actual_patient_nums, num_time_visits))

            # Train test split
            x_train, x_test, y_train, y_test = train_test_split(x_reshape, y_reshape, test_size=0.2, random_state=0)

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(64, input_shape=(num_time_visits, num_features)))
            model.add(Dense(num_time_visits, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Define callbacks for early stopping and checkpoint
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model_checkpoint_path = './lstm_training/' + column_name + '_model_checkpoint.h5'

            checkpoint = ModelCheckpoint(
                filepath=model_checkpoint_path,
                monitor='val_loss', 
                save_best_only=True)
            
            # Train the model
            history = model.fit(x_reshape, y_reshape, validation_split=0.2, epochs=50, batch_size=16, callbacks=[early_stop, checkpoint])

            # Plot learning curve based on history
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model Learning Curve')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.show()
            
            # Evaluate the model on the test set
            y_test_pred = model.predict(x_test)
            y_test_pred_binary = np.round(y_test_pred).astype(int)
            accuracy_test_set = accuracy_score(y_test.flatten(), y_test_pred_binary.flatten())
            print('TestSet Accuracy:', accuracy_test_set)
            plot_confusion_matrix(y_test.flatten(), y_test_pred.flatten(), ['Negative', 'Positive'])
            auc_test_set = plot_roc(y_test.flatten(), y_test_pred.flatten())

            # Evaluate the model on the train set
            y_train_pred = model.predict(x_train)
            y_train_pred_binary = np.round(y_train_pred).astype(int)
            accuracy_train_set = accuracy_score(y_train.flatten(), y_train_pred_binary.flatten())
            print('TrainSet Accuracy:', accuracy_train_set)
            plot_confusion_matrix(y_train.flatten(), y_train_pred.flatten(), ['Negative', 'Positive'])
            auc_train_set = plot_roc(y_train.flatten(), y_train_pred.flatten())

            # Evaluate the model on the whole dataset
            y_pred = model.predict(x_reshape)
            y_pred_binary = np.round(y_pred).astype(int)
            accuracy_data_set = accuracy_score(y_reshape.flatten(), y_pred_binary.flatten())
            print('DatatSet Accuracy:', accuracy_data_set)
            plot_confusion_matrix(y_reshape.flatten(), y_pred.flatten(), ['Negative', 'Positive'])
            auc_data_set = plot_roc(y_reshape.flatten(), y_pred.flatten())

            single_column_propensity_score_df = pd.DataFrame(y_pred.flatten())
    elif model_name == "LR": 
        profiles.groupby(column_name).mean() #calculating mean as per biguanide prescription
        x, y = profiles.drop(columns = [column_name]), profiles[column_name]
        normalize(x, x.columns)
        x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y, test_size=0.2, random_state=0)
        # Logistic Regression Model Fitting
        logistic_regression = LogisticRegression()
        logistic_regression.fit(x, y)
        logistic_regression.fit(x_train, y_train)

        # Print accuracy_score and plot confusion_matrix, , roc_curve, auc on test_set
        ps_score_estimate_test_set = logistic_regression.predict_proba(x_test)
        column_df = pd.DataFrame(ps_score_estimate_test_set)
        column_real_df_test_set = column_df.drop(columns=[0])
        accuracy_test_set = accuracy_score(y_test, column_real_df_test_set.round())
        auc_test_set = plot_roc(y_test, column_real_df_test_set)
        plot_confusion_matrix(y_test, column_real_df_test_set, ['Negative', 'Positive'])
        print('Test_Set Accuracy:', accuracy_test_set)

        # Print accuracy_score and plot confusion_matrix, , roc_curve, auc on train_set
        ps_score_estimate_train_set = logistic_regression.predict_proba(x_train)
        column_df = pd.DataFrame(ps_score_estimate_train_set)
        column_real_df_train_set = column_df.drop(columns=[0])
        accuracy_train_set = accuracy_score(y_train, column_real_df_train_set.round())
        auc_train_set = plot_roc(y_train, column_real_df_train_set)
        plot_confusion_matrix(y_train, column_real_df_train_set, ['Negative', 'Positive'])
        print('Train_Set Accuracy:', accuracy_train_set)

        # Print accuracy_score and plot confusion_matrix, , roc_curve, auc on dataset
        ps_score_estimate_dataset = logistic_regression.predict_proba(x)
        column_df = pd.DataFrame(ps_score_estimate_dataset)
        column_real_df_dataset = column_df.drop(columns=[0])
        accuracy_data_set = accuracy_score(y, column_real_df_dataset.round())
        auc_data_set = plot_roc(y, column_real_df_dataset)
        plot_confusion_matrix(y, column_real_df_dataset, ['Negative', 'Positive'])
        print('Dataset Accuracy:', accuracy_data_set)
        
        # Histogram (for PS overlap)
        profiles['ps'] = ps_score_estimate_dataset[:, 1]
        sns.histplot(data=profiles, x='ps', hue=column_name)
        plt.show() 

        return column_real_df_dataset, accuracy_test_set, accuracy_train_set, accuracy_data_set, auc_test_set, auc_train_set, auc_data_set
    
    return single_column_propensity_score_df, accuracy_test_set, accuracy_train_set, accuracy_data_set, auc_test_set, auc_train_set, auc_data_set 