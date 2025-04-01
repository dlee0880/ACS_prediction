import warnings
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from models import  data_loading, spliting_scaling, training_model, evaluate_cnn_on_datasets

# Initial settings
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams.update({'pdf.fonttype': 'truetype'})
warnings.simplefilter('ignore')
plt.style.use("fivethirtyeight")

def main(model_name):
    if model_name == "CNN":
        results_cnn = evaluate_cnn_on_datasets(datasets)
        print(results_cnn)
    else:
        data_path = {
            "XGBOOST": '',
            "DNN": '',
            "LGBM": '',
            "LR": '',
            "SVM": '',
            "RF": ''
        }

        label_col = {
            "XGBOOST": 'acs_label',
            "DNN": 'acs_label',
            "LGBM": 'Y',
            "LR": 'acs_label',
            "SVM": 'acs_label',
            "RF": 'acs_label'
        }
        # Data loading
        X, y = data_loading(data_path, label_col, model_name)

        # Data Spliting & Scaling
        X_train, X_test, y_train, y_test = spliting_scaling(model_name, X, y)

        if model_name == "RF":
            RFC_best = training_model(model_name,  X_train, X_test, y_train, y_test)

            data = pd.read_csv(data_path)

            # To view the feature scores, creating a seaborn bar plot
            feature_scores = pd.Series(RFC_best.feature_importances_, index=X_train.columns).sort_values(ascending=False)
            f, ax = plt.subplots(figsize=(30, 24))
            ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=data)
            ax.set_title("Visualize feature scores of the features")
            ax.set_yticklabels(feature_scores.index)
            ax.set_xlabel("Feature importance score")
            ax.set_ylabel("Features")
            plt.show()

        else:
            training_model(model_name,  X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    model_name = input("Model Name [LGBM/XGBOOST/LR/SVM/RF/DNN]: ")
    main(model_name)
