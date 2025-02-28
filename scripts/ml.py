import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from logger import Logger
from sklearn.model_selection import cross_validate


class Ml:

    def __init__(self) -> None:
        """Initilize class."""
        try:
            pass
            # self.logger = Logger("ml.log").get_app_logger()
            # self.logger.info(
            # 'Successfully Instantiated ml Class Object')
        except Exception:
            # self.logger.exception(
            # 'Failed to Instantiate Preprocessing Class Object')
            sys.exit(1)

    def cross_validation(self, model, _X, _y, _cv=5):
        """Perform 5 Folds Cross-Validation Parameters.
        ----------
        model: Python Class, default=None
                This is the machine learning algorithm to be used for training.
        _X: array
            This is the matrix of features.
        _y: array
            This is the target variable.
        _cv: int, default=5
            Determines the number of folds for cross-validation.
        Returns
        -------
        The function returns a dictionary containing the metrics 'accuracy', 'precision',
        'recall', 'f1' for both training set and validation set.
        """
        _scoring = ['accuracy', 'precision', 'recall', 'f1']
        coef = []
        results = cross_validate(estimator=model,
                                 X=_X,
                                 y=_y,
                                 cv=_cv,
                                 scoring=_scoring,
                                 return_train_score=True, return_estimator=True)

        # Print the coefficients of the features in the decision tree
        # print(results['estimator'].feature_importances_)
        # print("Coefficients: \n", results.best_estimator_.feature_importances_)

        # for model in results['estimator']:
        #     print(model.coef_)
        return {"Training Accuracy scores": results['train_accuracy'],
                "Mean Training Accuracy": results['train_accuracy'].mean()*100,
                "Training Precision scores": results['train_precision'],
                "Mean Training Precision": results['train_precision'].mean(),
                "Training Recall scores": results['train_recall'],
                "Mean Training Recall": results['train_recall'].mean(),
                "Training F1 scores": results['train_f1'],
                "Mean Training F1 Score": results['train_f1'].mean(),
                "Validation Accuracy scores": results['test_accuracy'],
                "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
                "Validation Precision scores": results['test_precision'],
                "Mean Validation Precision": results['test_precision'].mean(),
                "Validation Recall scores": results['test_recall'],
                "Mean Validation Recall": results['test_recall'].mean(),
                "Validation F1 scores": results['test_f1'],
                "Mean Validation F1 Score": results['test_f1'].mean(),
                "Coefficients": coef
                }

    # Grouped Bar Chart for both training and validation data

    def plot_result(self, x_label, y_label, plot_title, train_data, val_data, image_name):
        """Plot a grouped bar chart showing the training and validation results of the ML model in each fold after applying K-fold cross-validation.
         Parameters
         ----------
         x_label: str,
            Name of the algorithm used for training e.g 'Decision Tree'
         y_label: str,
            Name of metric being visualized e.g 'Accuracy'
         plot_title: str,
            This is the title of the plot e.g 'Accuracy Plot'
         train_result: list, array
            This is the list containing either training precision, accuracy, or f1 score.
         val_result: list, array
            This is the list containing either validation precision, accuracy, or f1 score.
         Returns
         -------
         The function returns a Grouped Barchart showing the training and validation result
         in each fold.
        """

        # Set size of plot
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.savefig(image_name)
        # plt.show()