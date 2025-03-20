import os
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from models import single_column_train_and_estimate_ps_score
from utils import export_csv

# Initial settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'pdf.fonttype': 'truetype'})
warnings.simplefilter('ignore')

def main(model_name):
    parent_dir_dict = {
        "XGBOOST": "C:/Users/CCADD/Documents/GitHub/final_propensity_acs_project_XGBoost/models/",
        "LGBM": ".../final_propensity_acs_project_LGBM/",
        "DNN": "",
        "LSTM": "",
        "LR": ""
    }

    data_path = {
    "LSTM": 'C:/Users/CCADD/Documents/GitHub/final_propensity_acs_project_LSTM - Copy/dataset/propensity_score_estimate/new_basetable_ps_timestep.csv',
    "DNN": 'C:/Users/CCADD/Documents/GitHub/final_propensity_acs_project_DNN - Copy/dataset/propensity_score_estimate/final_dataset_collapsed.csv',
    "XGBOOST": 'C:/Users/CCADD/Documents/GitHub/final_propensity_acs_project_XGBoost/dataset/propensity_score_estimate/final_dataset_collapsed.csv',
    "LR": 'C:/Users/CCADD/Documents/GitHub/final_propensity_acs_project_LR/dataset/propensity_score_estimate/final_dataset_collapsed.csv',
    "LGBM": '.../final_dataset_collapsed.csv'
    }

    binary_categorical_columns_dict = {
    "LSTM": [
        'a10ba', 'a10bb', 'a10bf', 'a10bg', 'a10bh', 'a01ad', 'a02ba', 'a03ax', 
        'a03fa', 'b01ac', 'c01bb', 'c01ca', 'c01da', 'c03aa', 'c03ca', 'c03da', 'c07aa', 'c07ab', 'c07ag', 
        'c08ca', 'c08db', 'c09aa', 'c09ca', 'c10aa', 'c22', 'e78', 'h25', 'h36', 'i10', 'i20', 'i63', 'k74', 
        'n02ab', 'n18', 'r69'
        ],
        "DNN": [
        'a10ba', 'a10bb', 'a10bf', 'a10bg', 'a10bh', 'a01ad', 'a02ba', 'a03ax', 'a03fa', 'b01ac', 
            'c01bb', 'c01ca', 'c01da', 'c03aa', 'c03ca', 'c03da', 'c07aa', 'c07ab', 'c07ag', 'c08ca', 
            'c08db', 'c09aa', 'c09ca', 'c10aa', 'n02ab', 'c22', 'e78', 'h25', 'h36', 'i10', 'i20', 
            'i63', 'k74', 'n18', 'r69'
        ],
        "XGBOOST": [
        'a10ba', 'a10bb', 'a10bf', 'a10bg', 'a10bh', 'a01ad', 'a02ba', 'a03ax', 
        'a03fa', 'b01ac', 'c01bb', 'c01ca', 'c01da', 'c03aa', 'c03ca', 'c03da', 'c07aa', 'c07ab', 'c07ag', 
        'c08ca', 'c08db', 'c09aa', 'c09ca', 'c10aa', 'c22', 'e78', 'h25', 'h36', 'i10', 'i20', 'i63', 'k74', 
        'n02ab', 'n18', 'r69'
        ],
        "LR": [
        'a10ba', 'a10bb', 'a10bf', 'a10bg', 'a10bh', 'a01ad', 'a02ba', 'a03ax', 
        'a03fa', 'b01ac', 'c01bb', 'c01ca', 'c01da', 'c03aa', 'c03ca', 'c03da', 'c07aa', 'c07ab', 'c07ag', 
        'c08ca', 'c08db', 'c09aa', 'c09ca', 'c10aa', 'c22', 'e78', 'h25', 'h36', 'i10', 'i20', 'i63', 'k74', 
        'n02ab', 'n18', 'r69'
        ],
        "LGBM": [
        'a10ba', 'a10bb', 'a10bf', 'a10bg', 'a10bh', 'a01ad', 'a02ba', 'a03ax', 
        'a03fa', 'b01ac', 'c01bb', 'c01ca', 'c01da', 'c03aa', 'c03ca', 'c03da', 'c07aa', 'c07ab', 'c07ag', 
        'c08ca', 'c08db', 'c09aa', 'c09ca', 'c10aa', 'c22', 'e78', 'h25', 'h36', 'i10', 'i20', 'i63', 'k74', 
        'n02ab', 'n18', 'r69'
        ]
    }

    #Data Loading
    profiles = pd.read_csv(data_path[model_name])
    dataframe = pd.read_csv(data_path[model_name])
    results_df = pd.DataFrame()
    binary_categorical_columns = binary_categorical_columns_dict[model_name]


    #Initialize the total values
    total_accuracy_test_set = 0
    total_accuracy_train_set = 0
    total_accuracy_data_set = 0
    total_auc_test_set = 0
    total_auc_train_set = 0
    total_auc_data_set = 0

    # Initialize first column of results_df
    single_column_result_df = pd.DataFrame({'':['test_accuracy', 'train_accuracy', 'accuracy_data_set', 'auc_test_set', 'auc_train_set', 'auc_data_set']})
    results_df = pd.concat([results_df, single_column_result_df], axis=1)

    # For loop to automate estimate ps_score for all binary categorical columns
    for binary_categorical in binary_categorical_columns:
        print("/***************" + binary_categorical + "******************/" )
        single_column_propensity_score_df, accuracy_test_set, accuracy_train_set, accuracy_data_set, auc_test_set, auc_train_set, auc_data_set = single_column_train_and_estimate_ps_score(model_name, column_name, profiles, parent_dir_dict)
        single_column_propensity_score_df.rename(columns={single_column_propensity_score_df.columns[0]: binary_categorical}, inplace=True)
        dataframe[binary_categorical] = single_column_propensity_score_df[binary_categorical]
        
        # Add single_column_df to results_df
        single_column_result_df = pd.DataFrame({binary_categorical:[accuracy_test_set, accuracy_train_set, accuracy_data_set, auc_test_set, auc_train_set, auc_data_set]})
        results_df = pd.concat([results_df, single_column_result_df], axis=1)
        
        # calculate total results
        total_accuracy_test_set = total_accuracy_test_set + accuracy_test_set
        total_accuracy_train_set = total_accuracy_train_set + accuracy_train_set
        total_accuracy_data_set = total_accuracy_data_set + accuracy_data_set
        total_auc_test_set = total_auc_test_set + auc_test_set
        total_auc_train_set = total_auc_train_set + auc_train_set
        total_auc_data_set = total_auc_data_set + auc_data_set

    # Calculate mean results values
    mean_accuracy_test_set = total_accuracy_test_set/35
    mean_accuracy_train_set = total_accuracy_train_set/35
    mean_accuracy_data_set = total_accuracy_data_set/35
    mean_auc_test_set = total_auc_test_set/35
    mean_auc_train_set = total_auc_train_set/35
    mean_auc_data_set = total_auc_data_set/35

    # Concat mean results values
    single_column_result_df = pd.DataFrame({'mean':[mean_auc_test_set, mean_auc_train_set, mean_auc_data_set, mean_accuracy_test_set, mean_accuracy_train_set, mean_auc_data_set]})
    results_df = pd.concat([results_df, single_column_result_df], axis=1)

    export_csv(dataframe, results_df, model_name)


if __name__ == "__main__":
    model_name = input()
    
    if model_name == "LSTM": # Make directory to save lstm model checkpoint
        try:
            new_folder = "lstm_training"
            os.mkdir(new_folder)
        except FileExistsError:
            print("Folder already exists")

    main(model_name)
    