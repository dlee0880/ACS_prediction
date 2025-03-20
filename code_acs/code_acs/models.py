import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
import lightgbm as lgb
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import keras_tuner as kt
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, BatchNormalization, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score
from my_dir.code_acs.utils import plot_confusion_matrix, print_score, plot_pre_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

################
# Data Loading #
################

def data_loading(data_path, label_col, model_name): 
    """Data loading
    Args:
        model_name (str): Name of the models; XGBOOST, DNN, LGBM, LR, SVM, RF
    Returns 
        X: features
        y: label
    """

    data = pd.read_csv(data_path[model_name])
    X = data.drop(column = label_col[model_name])
    y = data[label_col[model_name]]
    return X, y

############################
# Data Splitting & Scaling #
############################

def spliting_scaling(model_name, X, y, test_size = 0.2, scaler = MinMaxScaler()):
    """SMOTE & Data Spliting & Scaling
    Args:
        model_name (str): Name of the models; XGBOOST, DNN, LGBM, LR, SVM, RF
        test_size (float): The ratio of test set; default = 0.2
        scaler: sklearn preproecessing tool; default = MinMaxScaler
    Returns:
        X_train, X_test, y_train, y_test  
    """
    random_state_split = {
        "LGBM": 43,
        "DNN": 111993,
        "XGBOOST": 14256656,
        "LR": 31332,
        "SVM": 0,
        "RF": 0
    }
        
    if model_name in ['XGBOOST', 'LGBM', 'LR','DNN']:
        # SMOTE
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, 
                                                        random_state= random_state_split[model_name])
    if model_name in ['XGBOOST', 'LGBM', 'LR','SVM']:
        
        # Feature scaling
        cols = X_train.columns

        # Scaling
        scaler = scaler
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=[cols])
        X_test = pd.DataFrame(X_test, columns=[cols])

    if model_name == "DNN":
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)

    return X_train, X_test, y_train, y_test

#########################
# HYPERPARAMETER TUNING #
#########################

def tune_hyperparameter(model_name, X_train, y_train):
    """Hyperparameter tuning
    Args:
        model_name (str): 
        X_train, y_train (DataFrame): training dataset
    Returns:
        mdl: model
        grid.best_params_: best hyperparameter 
    """
    if model_name in ['XGBOOST', 'LGBM']:
        X_train_np = X_train.to_numpy()
        y_train_np = y_train.to_numpy()

        if model_name == 'XGBOOST':
            # Declaring parameters
            params = {
                'objective': 'binary:logistic',
                'max_depth': 4,
                'lambda': 10,
                'learning_rate': 1.0,
                'n_estimators': 100
            }
            # Parameters to optimize
            params_grid = {
                'max_depth': [2, 4, 6, 8],
                'lambda': [1, 2, 3, 4, 5, 6],
                'learning_rate': [0.2, 0.4, 0.6, 0.8, 1.0],
                'n_estimators' : [100, 150, 200, 250, 300]
            }

            mdl = XGBClassifier(objective= 'binary:logistic',
                    learning_rate = params['learning_rate'],
                    n_estimators = params['n_estimators'], 
                    max_depth = params['max_depth'],
                    reg_lambda = params['lambda'],
                    seed=1236757)

        if model_name == 'LGBM': 
            # Declaring parameters
            params = {
                'max_depth' : -1,
                'num_leaves': 64,
                'learning_rate': 0.07,
                'max_bin': 512,
                'subsample_for_bin': 200,
                'subsample': 1,
                'subsample_freq': 1,
                'colsample_bytree': 0.8,
                'num_class' : 1,
                'metric' : 'binary_error',
                'verbosity' : 0,
                'reg_lambda': 0.5
            }

            # Parameters to optimize
            params_grid = {
                'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'n_estimators': [8, 16, 32, 64, 128, 256],
                'num_leaves': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
                'colsample_bytree' : [0.60, 0.65],
                'reg_lambda' : [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            }

            mdl = lgb.LGBMClassifier(
                boosting_type= 'gbdt',
                objective = 'binary',
                n_jobs = 5, 
                max_depth = params['max_depth'],
                max_bin = params['max_bin'],
                subsample_for_bin = params['subsample_for_bin'],
                subsample = params['subsample'],
                subsample_freq = params['subsample_freq'],
                learning_rate = params_grid['learning_rate'],
                n_estimators = params_grid['n_estimators'],
                num_leaves = params_grid['num_leaves'],
                colsample_bytree = params_grid['colsample_bytree'],
                reg_lambda = params_grid['reg_lambda']
            )
        
        #create the grid
        grid = GridSearchCV(mdl, params_grid, verbose=1, cv=10, n_jobs=-1)
        #Search for optimal parameters
        grid.fit(X_train_np, y_train_np)

    elif model_name == 'LR':
        param_grid = {
            "penalty": ['None', 'l1', 'l2', 'elasticnet'], 
            "C": [0.001, 0.01, 0.1, 0.5, 0.6, 0.7, 0.8, 1, 10, 100, 1000], # inverse of regularization strength, default = 1 
            "class_weight": ['balanced'] , 
            "solver": ['liblinear', 'saga', 'newton-cg', 'sag', 'lbfgs']
        }

        mdl = LogisticRegression(solver='liblinear')

        grid = GridSearchCV(estimator=mdl, param_grid=param_grid, scoring='f1', verbose=1, cv=10, n_jobs=-1)
        grid.fit(X_train, y_train)
        
    elif model_name == 'SVM':
        params_grid = [ 
            {'C':[1, 10, 100, 1000], 'kernel':['linear']},
            {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            {'C':[1, 10, 100, 1000], 'kernel':['poly'], 'degree': [2,3,4] ,'gamma':[0.01,0.02,0.03,0.04,0.05]},
            {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'], 'degree': [2,3,4]} 
        ]

        # default hyperparameters: kernel=rbf, C=1.0 and gamma=auto
        mdl=SVC() 

        grid = GridSearchCV(mdl, params_grid, verbose=0, cv = 10, scoring = 'accuracy')
        grid.fit(X_train, y_train)

        return mdl

    elif model_name == "RF":
        params_grid = {
            "max_features": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "min_samples_split": [ 2, 3, 5],
            "min_samples_leaf": [2, 3, 5],
            "n_estimators": [10, 50, 100, 150, 200],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None],
            "bootstrap": [True]
        }
        
        kfold = StratifiedKFold(n_splits=10)
        mdl = RandomForestClassifier(random_state=1)
        gsRFC = GridSearchCV(mdl, param_grid = params_grid, cv=kfold, scoring="accuracy")
        gsRFC.fit(X_train, y_train)
        RFC_best = gsRFC.best_estimator_

        return RFC_best
    
    elif model_name == "DNN":
 
        random_search_params = {
            'num_dense_layers_min': 1,
            'num_dense_layers_max': 6,
            'dense_dims': [8, 16, 32, 64, 128],
            'activation': ['relu', 'sigmoid', 'softmax', 'tanh'],
            'dropout_rate_min': 0.1,
            'dropout_rate_max': 0.9
        }

        input_shape=keras.Input(shape=(X_train.shape[1],))

        grid_search_params = random_search_dnn(random_search_params, input_shape, X_train, y_train, 'acs_label')
        print('acs_label' + " ---> grid_search_params:")
        print(grid_search_params)

        # grid search for hyperparameter
        model_dense = grid_search_dnn(grid_search_params, input_shape, X_train, y_train, 'acs_label')

        return model_dense
    
    return mdl, grid.best_params_

############
# TRAINING #
############

def training_model(model_name, X_train, X_test, y_train, y_test):
    if model_name == "LGBM":
        # Training
        _, params = tune_hyperparameter(model_name, X_train, y_train)
        train_data = lgb.Dataset(X_train.to_numpy(), label=y_train.to_numpy())
        lgbm = lgb.train(params, train_data, num_boost_round=200, verbose_eval= 4)

        # Evaluating
        y_pred = lgbm.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy: %f' % accuracy)
        precision = precision_score(y_test, y_pred)
        print('Precision: %f' % precision)
        recall = recall_score(y_test, y_pred)
        print('Recall: %f' % recall)
        f1 = f1_score(y_test, y_pred)
        print('F1 score: %f' % f1)
        auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC: %f' % auc)
        print(classification_report(y_test, y_pred))
        print_score(lgbm, X_train, y_train, X_test, y_test, train=False)

    elif model_name == 'XGBOOST':
        # Training
        _, params = tune_hyperparameter(model_name, X_train, y_train)

        xgb_reg = XGBClassifier(**params)
        xgb_reg.fit(X_train, y_train)

        # Evaluating
        y_pred = xgb_reg.predict(X_test)

        print_score(xgb_reg, X_train, y_train, X_test, y_test, train=False)
        auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC: %f' % auc)

    elif model_name == "LR":
        # Training
        _, params = tune_hyperparameter(model_name, X_train, y_train)
        lr_clf = LogisticRegression(**params)
        lr_clf.fit(X_train, y_train)

        # Evaluating
        y_pred = lr_clf.predict(X_test)

        plot_pre_curve(y_test,y_pred)

    elif model_name == 'SVM':
        # Training
        mdl = tune_hyperparameter(model_name, X_train, y_train)
        mdl.fit(X_train, y_train)

        # Evaluating
        y_pred = mdl.predict(X_test)

        accuracy_test_set = accuracy_score(y_test, y_pred)
        print('TestSet Accuracy:', accuracy_test_set)
        plot_confusion_matrix(y_test, y_pred, ['Negative', 'Positive'])
        print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
        print_score(mdl, X_train, y_train, X_test, y_test, train=False)

        #Check for overfit: print the scores on training and test set and if the two scores are comparable, no questino of overfit
        print('Training set score: {:.4f}'.format(mdl.score(X_train, y_train)))
        print('Test set score: {:.4f}'.format(mdl.score(X_test, y_test)))

    elif model_name == "RF":
        # Training
        RFC_best = tune_hyperparameter(model_name, X_train, y_train)

        # Evaluating
        y_pred = RFC_best.predict(X_test)

        accuracy_test_set = accuracy_score(y_test, y_pred)
        print('TestSet Accuracy:', accuracy_test_set)
        plot_confusion_matrix(y_test, y_pred, ['Negative', 'Positive'])
        print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
        
        # Print score -> classification report
        print_score(RFC_best, X_train, y_train, X_test, y_test, train=False)
        precision = precision_score(y_test, y_pred)
        print('Precision: %f' % precision)
        auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC: %f' % auc)

        return RFC_best


    elif model_name == 'DNN':
        # Training
        model_dense = tune_hyperparameter(model_name, X_train, y_train)
        history = model_dense.fit(X_train, y_train, epochs = 400, validation_split=0.2, verbose=1)

        # plot the history
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
    
        # Evaluating
        y_pred = model_dense.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        print('ROC AUC: %f' % auc)
        print_score(model_dense, X_train, y_train, X_test, y_test, train=False)
        [loss, accuracy] = model_dense.evaluate(X_test, y_test, verbose=0)
        [loss, accuracy][1:]

###############################
# HP TUNING FUNCTIONS FOR DNN #
################################

def build_model_dnn(hp, param_grids, input_shape):
    num_dense_layers = hp.Int ('num_dense_layers', min_value=param_grids['num_dense_layers_min'], max_value=param_grids['num_dense_layers_max'])
    dense_dims = [hp.Choice('dense_dims', param_grids['dense_dims']) for i in range(num_dense_layers)]
    activation = hp.Choice('activation', param_grids['activation'])
    dropout_rate = hp.Float('dropout_rate', min_value=param_grids['dropout_rate_min'], max_value=param_grids['dropout_rate_max'], step=0.1)
    dnn_x = input_shape
    for i in range(num_dense_layers):
        dnn_x = Dense(dense_dims[i], activation=activation)(dnn_x)
        dnn_x = BatchNormalization()(dnn_x)
        dnn_x = Dropout(dropout_rate)(dnn_x)
    prob_outcome = Dense(1, activation='relu')(dnn_x)
    model_dense = keras.Model(inputs=input_shape, outputs=prob_outcome)
    model_dense.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy']) 
    return model_dense

def random_search_dnn(param_grids, input_shape, X_train, y_train,column_name):
    """ Random search for 100 trials
    Args:
        param_grids (dictionary): A dictionary to storage searching space for DNN architecture
    Returns:
        grid_search_params: A dictionary of searching space for next grid search step
    """
    tuner = kt.RandomSearch(lambda hp: build_model_dnn(hp, param_grids, input_shape), 
                            objective='val_accuracy', 
                            max_trials=50, 
                            project_name='random_search_project_' + column_name)
    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_train, y_train), callbacks=[stop_early])
    model_dense = tuner.get_best_models()[0]
    model_dense.summary()
    grid_search_params_json = tuner.oracle.get_best_trials(num_trials=50)[0].hyperparameters.values
    grid_search_params = {
        'num_dense_layers_min': max(grid_search_params_json['num_dense_layers'] - 1, 1),
        'num_dense_layers_max': max(grid_search_params_json['num_dense_layers'] + 1, 0),
        'dense_dims': [grid_search_params_json['dense_dims']],
        'activation': ['relu', 'sigmoid', 'softmax', 'tanh'],
        'dropout_rate_min': max(grid_search_params_json['dropout_rate'] - 0.1, 0),
        'dropout_rate_max': max(grid_search_params_json['dropout_rate'] + 0.1, 0)
    }
    return grid_search_params

def grid_search_dnn(param_grids, input_shape, X_train, y_train, column_name):
    """Grid search for 27 trials
    Args:
        param_grids (dictionary): A dictionary to storage searching space for DNN architecture
    Returns:
        model_dense: Keras model storage learned weights
    """
    #Hyperparameter Optimization
    tuner = kt.GridSearch(lambda hp: build_model_dnn(hp, param_grids, input_shape), 
                          objective='val_accuracy', 
                          max_trials=27, 
                          project_name='grid_search_project_' + column_name)
    stop_early = EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, batch_size=64, epochs=200, validation_data=(X_train, y_train), callbacks=[stop_early])
    model_dense = tuner.get_best_models()[0]
    model_dense.summary()
    hps = tuner.oracle.get_best_trials(num_trials=27)[0].hyperparameters.values
    print(column_name + ' ---> HyperParameters: {}'.format(hps))
    
    return model_dense