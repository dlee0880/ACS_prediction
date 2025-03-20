import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from tabulate import tabulate
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression

###############
# Path define #
###############

# FINAL_PS_LR_DATASET
FINAL_PS_LR_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/propensity_acs_project/datasets/acs_prediction/ps_score_dataset/final_ps_LR.csv'
OUTPUT_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LR_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_LR_dataset/OUTPUT_LR_acs_prediction_on_final_ps_score_LR_dataset.csv'
ACCURACY_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LR_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_LR_dataset/ACCURACY_LR_acs_prediction_on_final_ps_score_LR_dataset.csv'

# FINAL_PS_LGBM_DATASET
FINAL_PS_LGBM_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/propensity_acs_project/datasets/acs_prediction/ps_score_dataset/final_ps_LGBM.csv'
OUTPUT_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LGBM_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_LGBM_dataset/OUTPUT_LR_acs_prediction_on_final_ps_score_LGBM_dataset.csv'
ACCURACY_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LGBM_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_LGBM_dataset/ACCURACY_LR_acs_prediction_on_final_ps_score_LGBM_dataset.csv'

# FINAL_PS_XGBOOST_DATASET
FINAL_PS_XGBOOST_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/propensity_acs_project/datasets/acs_prediction/ps_score_dataset/final_ps_XGBoost.csv'
OUTPUT_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_XGBOOST_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_XGBoost_dataset/OUTPUT_LR_acs_prediction_on_final_ps_score_XGBoost_dataset.csv'
ACCURACY_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_XGBOOST_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_XGBoost_dataset/ACCURACY_LR_acs_prediction_on_final_ps_score_XGBoost_dataset.csv'

# FINAL_PS_DNN_DATASET
FINAL_PS_DNN_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/propensity_acs_project/datasets/acs_prediction/ps_score_dataset/final_ps_DNN.csv'
OUTPUT_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_DNN_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_DNN_dataset/OUTPUT_LR_acs_prediction_on_final_ps_score_DNN_dataset.csv'
ACCURACY_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_DNN_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_DNN_dataset/ACCURACY_LR_acs_prediction_on_final_ps_score_DNN_dataset.csv'


# FINAL_P0_LSTM_DATASET
FINAL_PS_LSTM_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/propensity_acs_project/datasets/acs_prediction/ps_score_dataset/final_ps_LSTM.csv'
OUTPUT_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LSTM_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_LSTM_dataset/OUTPUT_LR_acs_prediction_on_final_ps_score_LSTM_dataset.csv'
ACCURACY_LR_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LSTM_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/ps_score_dataset/final_ps_LSTM_dataset/ACCURACY_LR_acs_prediction_on_final_ps_score_LSTM_dataset.csv'


# BINARY_DATASET_PATH
BINARY_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/propensity_acs_project/datasets/acs_prediction/binary_dataset/final_dataset_collapsed - acs_label.csv'
OUTPUT_LR_ACS_PREDICTION_RESULT_ON_BINARY_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/binary_dataset/OUTPUT_LR_acs_prediction_result_on_binary_dataset.csv'
ACCURACY_LR_ACS_PREDICTION_RESULT_ON_BINARY_DATASET_PATH = 'C:/Users/CCADD/Documents/GitHub/propensity_acs_project_10185/raw_experiment_results/acs_prediction/binary_dataset/ACCURACY_LR_acs_prediction_result_on_binary_dataset.csv'


# Initial setting
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'pdf.fonttype': 'truetype'})
warnings.simplefilter('ignore')


###
#Visualization and helper functions
###
def plot_roc(y_test, y_hat_prob):
    plt.figure(figsize = (8, 8))
    false_positive_rate, recall, thresholds = roc_curve(y_test, y_hat_prob)
    roc_auc = auc(false_positive_rate, recall)
    plt.title('Receiver Operating Characteristics (ROC)')
    plt.plot(false_positive_rate, recall, 'r', label = 'AUC = %0.3f' % roc_auc)
    plt.fill_between(false_positive_rate, recall, color = 'r', alpha = 0.025)
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.plot([0.1], [0.1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.show()
    return roc_auc

def plot_confusion_matrix(y, y_hat_prob, labels):
    y_hat = np.where(y_hat_prob > 0.5, 1, 0)
    cm = confusion_matrix(y, y_hat)
    accuracy = accuracy_score(y, y_hat)
    plt.figure(figsize = (4, 4))
    sns.heatmap(cm, xticklabels = labels, yticklabels = labels,
                annot = True, cbar = False, fmt = 'd', annot_kws = {'size': 16},
                cmap = 'Wistia', vmin = 0.2)
    plt.title(f'Confusion Matrix\n({len(y)} samples, accuracy {accuracy:.3f})')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()

def print_df(df, col_width = 10, rows = 10, max_cols = 10):
    def short_srt(x):
        return x if len(x) < col_width else x[:col_width-3] + "..."
    df_short = df.head(rows).applymap(lambda x: short_srt(str(x)))

    if len(df_short.columns) > max_cols:
        df_short = df_short.iloc[:, 0:max_cols-1]
        df_short['...'] = '...'

    print(tabulate(df_short, headers='keys', tablefmt='psql'))
    print(f'{len(df)} rows x {len(df.columns)} columns')

def normalize(df, columns):
    for c in columns:
        df.loc[:, c] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())

###
#Data Loading
###

profiles = pd.read_csv(FINAL_PS_LSTM_DATASET_PATH)
dataframe = pd.read_csv(FINAL_PS_LSTM_DATASET_PATH)
results_df = pd.DataFrame()

# acs_label column for training and prediction
binary_categorical_columns = ['acs_label']


def single_column_train_and_estimate_ps_score(column_name):
    profiles.groupby(column_name).mean() #calculating mean as per biguanide prescription
    x, y = profiles.drop(columns = [column_name]), profiles[column_name]
    normalize(x, x.columns)
    x_train, x_test, y_train, y_test = train_test_split(x.to_numpy(), y, test_size=0.2, random_state=0)

    # Shuffle the x_train and y_train data together
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

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
    return single_column_propensity_score_df, accuracy_test_set, auc_test_set

#Initialize the total values

total_accuracy_test_set = 0
total_auc_test_set = 0

# Initialize first column of results_df
single_column_result_df = pd.DataFrame({'':['accuracy_test_set', 'auc_test_set']})
results_df = pd.concat([results_df, single_column_result_df], axis=1)

# For loop to automate estimate ps_score for all binary categorical columns
for binary_categorical in binary_categorical_columns:
    print("/***************" + binary_categorical + "******************/" )
    single_column_propensity_score_df, accuracy_test_set, auc_test_set = single_column_train_and_estimate_ps_score(binary_categorical)
    single_column_propensity_score_df.rename(columns={single_column_propensity_score_df.columns[0]: binary_categorical}, inplace=True)
    dataframe['predicted_probabilities'] = single_column_propensity_score_df[binary_categorical]

    # convert to binary and add to dataframe
    single_column_propensity_score_df['predicted_binary'] = (single_column_propensity_score_df[binary_categorical] >= 0.5).astype(int)
    dataframe = pd.concat([dataframe, single_column_propensity_score_df['predicted_binary']], axis=1)
    
    # Add single_column_df to results_df
    single_column_result_df = pd.DataFrame({binary_categorical:[accuracy_test_set, auc_test_set]})
    results_df = pd.concat([results_df, single_column_result_df], axis=1)
    
    # calculate total results
    total_accuracy_test_set = total_accuracy_test_set + accuracy_test_set
    total_auc_test_set = total_auc_test_set + auc_test_set



# Calculate mean results values
mean_accuracy_test_set = total_accuracy_test_set/len(binary_categorical_columns)
mean_auc_test_set = total_auc_test_set/len(binary_categorical_columns)

# Concat mean results values
single_column_result_df = pd.DataFrame({'mean':[mean_accuracy_test_set, mean_auc_test_set]})
results_df = pd.concat([results_df, single_column_result_df], axis=1)

# export to csv
dataframe.to_csv(OUTPUT_SVM_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LR_DATASET_PATH, index=False)
results_df.to_csv(ACCURACY_SVM_ACS_PREDICTION_RESULT_ON_FINAL_PS_SCORE_LR_DATASET_PATH, index=False)
