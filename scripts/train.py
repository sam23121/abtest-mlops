import os
import warnings
import sys
sys.path.append(os.path.abspath(os.path.join('../scripts')))

import pandas as pd
import numpy as np
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
    df.rename(columns = {'yes': 'clicked_or_not'}, inplace=True)

    # Get column names have less than 10 more than 2 unique values
    to_one_hot_encoding = [col for col in categorical_column if df[col].nunique() <= 10 and df[col].nunique() > 2]
    one_hot_encoded_columns = pd.get_dummies(df[to_one_hot_encoding])

    # Get Categorical Column names thoose are not in "to_one_hot_encoding"
    to_label_encoding = [col for col in categorical_column if not col in to_one_hot_encoding]
    le = LabelEncoder()
    df[to_label_encoding] = df[to_label_encoding].apply(le.fit_transform)

    df.drop(['date', 'browser'], axis=1, inplace=True)

    df = pd.concat([df, one_hot_encoded_columns], axis=1)

    y = df['clicked_or_not']
    X = df.drop(["clicked_or_not"], axis=1)

    return X, y




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # np.random.seed(40)

    # pd.set_option('max_column', None)
    df = pd.read_csv('../data/AdSmartABdata.csv', engine = 'python')

    X, y = pre_processing(df)
    

    
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

        predicted_qualities = model_pipeline.predict(X_test)
        acc_sco = accuracy_score(y_test, predicted_qualities)


        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        # print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # mlflow.log_param("alpha", alpha)
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
