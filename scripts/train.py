import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# To fill missing values
from sklearn.impute import SimpleImputer

# To Split our train data
from sklearn.model_selection import train_test_split


# To Train our data
# from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# from ml import ML


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def pre_processing(df):
    #droping the auction id since it has no value for the train
    df.drop('auction_id', axis=1, inplace=True)
    numerical_column = df.select_dtypes(exclude="object").columns.tolist()
    categorical_column = df.select_dtypes(include="object").columns.tolist()
    relevant_rows = df.query('yes == 1 | no == 1')


    df = relevant_rows.drop('no', axis=1)
    df.rename(columns = {'yes': 'brand_awareness'}, inplace=True)

    # Get column names have less than 10 more than 2 unique values
    to_one_hot_encoding = [col for col in categorical_column if df[col].nunique() <= 10 and df[col].nunique() > 2]
    one_hot_encoded_columns = pd.get_dummies(df[to_one_hot_encoding])

    # Get Categorical Column names thoose are not in "to_one_hot_encoding"
    to_label_encoding = [col for col in categorical_column if not col in to_one_hot_encoding]
    le = LabelEncoder()
    df[to_label_encoding] = df[to_label_encoding].apply(le.fit_transform)

    df.drop(['date', 'browser'], axis=1, inplace=True)

    df = pd.concat([df, one_hot_encoded_columns], axis=1)

    y = df['brand_awareness']
    X = df.drop(["brand_awareness"], axis=1)

    return X, y




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # np.random.seed(40)

    # pd.set_option('max_column', None)
    df = pd.read_csv('data/AdSmartABdata.csv', engine = 'python')

    X, y = pre_processing(df)
    
    axis_fs = 18 #fontsize
    title_fs = 22 #fontsize
    sns.set(style="whitegrid")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.23)

    with mlflow.start_run():

        # creating a pipeline
        model_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), ('model', LogisticRegression())])
        model_pipeline.fit(X_train, y_train)
        # lr = LogisticRegression()
        # lr.fit(X_train, y_train)
        train_score = model_pipeline.score(X_train, y_train)
        test_score = model_pipeline.score(X_test, y_test)

        with open("metrics.txt", 'w') as outfile:
            outfile.write("Training variance explained: %2.1f%%\n" % train_score)
            outfile.write("Test variance explained: %2.1f%%\n" % test_score)

        predicted_qualities = model_pipeline.predict(X_val)
        acc_sco = accuracy_score(y_val, predicted_qualities)


        (rmse, mae, r2) = eval_metrics(y_val, predicted_qualities)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("acc_sco", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # mlflow.sklearn.log_model(lr, "model")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model_pipeline, "model", registered_model_name="logisticRegAB")
        else:
            mlflow.sklearn.log_model(model_pipeline, "model")

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

        y_pred = model_pipeline.predict(X_test) + np.random.normal(0,0.25,len(y_test))
        y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
        res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

        ax = sns.scatterplot(x="true", y="pred",data=res_df)
        ax.set_aspect('equal')
        ax.set_xlabel('True predictions',fontsize = axis_fs) 
        ax.set_ylabel('Predicted predictions', fontsize = axis_fs)#ylabel
        ax.set_title('Residuals', fontsize = title_fs)

        # Make it pretty- square aspect ratio
        ax.plot([1, 10], [1, 10], 'black', linewidth=1)
        plt.ylim((2.5,8.5))
        plt.xlim((2.5,8.5))

        plt.tight_layout()
        plt.savefig("residuals_for_logesticregression.png",dpi=120)

    with mlflow.start_run():
        model_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), ('model', DecisionTreeClassifier(criterion = 'entropy'))])
        model_pipeline.fit(X_train, y_train)
        # lr = LogisticRegression()
        # lr.fit(X_train, y_train)
        train_score = model_pipeline.score(X_train, y_train)
        test_score = model_pipeline.score(X_test, y_test)

        with open("metrics2.txt", 'w') as outfile:
            outfile.write("Training variance explained: %2.1f%%\n" % train_score)
            outfile.write("Test variance explained: %2.1f%%\n" % test_score)

        predicted_qualities = model_pipeline.predict(X_val)
        acc_sco = accuracy_score(y_val, predicted_qualities)


        (rmse, mae, r2) = eval_metrics(y_val, predicted_qualities)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("cretrion", 'entropy')
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("acc_sco", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # mlflow.sklearn.log_model(lr, "model")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model_pipeline, "model", registered_model_name="decissiontreeAB")
        else:
            mlflow.sklearn.log_model(model_pipeline, "model")

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

        y_pred = model_pipeline.predict(X_test) + np.random.normal(0,0.25,len(y_test))
        y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
        res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

        ax = sns.scatterplot(x="true", y="pred",data=res_df)
        ax.set_aspect('equal')
        ax.set_xlabel('True predictions',fontsize = axis_fs) 
        ax.set_ylabel('Predicted predictions', fontsize = axis_fs)#ylabel
        ax.set_title('Residuals', fontsize = title_fs)

        # Make it pretty- square aspect ratio
        ax.plot([1, 10], [1, 10], 'black', linewidth=1)
        plt.ylim((2.5,8.5))
        plt.xlim((2.5,8.5))

        plt.tight_layout()
        plt.savefig("residuals_for_decisiontree.png",dpi=120)

        

    with mlflow.start_run():

        model_pipeline = Pipeline(steps=[('scaler', MinMaxScaler()), ('model', RandomForestClassifier(n_estimators = 10, criterion = 'entropy'))])
        model_pipeline.fit(X_train, y_train)
        # lr = LogisticRegression()
        # lr.fit(X_train, y_train)
        train_score = model_pipeline.score(X_train, y_train)
        test_score = model_pipeline.score(X_test, y_test)

        with open("metrics3.txt", 'w') as outfile:
            outfile.write("Training variance explained: %2.1f%%\n" % train_score)
            outfile.write("Test variance explained: %2.1f%%\n" % test_score)

        predicted_qualities = model_pipeline.predict(X_val)
        acc_sco = accuracy_score(y_val, predicted_qualities)


        (rmse, mae, r2) = eval_metrics(y_val, predicted_qualities)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", 10)
        mlflow.log_param("cretrion", 'entropy')
        mlflow.log_metric("train_score", train_score)
        mlflow.log_metric("acc_sco", train_score)
        mlflow.log_metric("test_score", test_score)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # mlflow.sklearn.log_model(lr, "model")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model_pipeline, "model", registered_model_name="randomforestAB")
        else:
            mlflow.sklearn.log_model(model_pipeline, "model")

        print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

        y_pred = model_pipeline.predict(X_test) + np.random.normal(0,0.25,len(y_test))
        y_jitter = y_test + np.random.normal(0,0.25,len(y_test))
        res_df = pd.DataFrame(list(zip(y_jitter,y_pred)), columns = ["true","pred"])

        ax = sns.scatterplot(x="true", y="pred",data=res_df)
        ax.set_aspect('equal')
        ax.set_xlabel('True predictions',fontsize = axis_fs) 
        ax.set_ylabel('Predicted predictions', fontsize = axis_fs)#ylabel
        ax.set_title('Residuals', fontsize = title_fs)

        # Make it pretty- square aspect ratio
        ax.plot([1, 10], [1, 10], 'black', linewidth=1)
        plt.ylim((2.5,8.5))
        plt.xlim((2.5,8.5))

        plt.tight_layout()
        plt.savefig("residuals_for_random forest.png",dpi=120) 
