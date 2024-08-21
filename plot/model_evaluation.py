# Description: This file contains the class for evaluating the model and plotting the metrics.

import numpy as np
import os
import tensorflow as tf
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import bokeh
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot


class Plot:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_dir = os.path.join(config['paths']['result'], config['experiment_name'])

    def plot_metrics(self, history):
        """
        Plot the training and validation metrics.

        :param history: A dictionary containing the training and validation metrics.
        """
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(history['mean_absolute_error'], label='Training MAE')
        plt.plot(history['val_mean_absolute_error'], label='Validation MAE')
        plt.legend(loc='lower right')
        plt.title('Training and Validation MAE')

        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.show()

    def evaluate_model(self, model, test_data: Dict[str, Any]):
        """
        Evaluate the model using the provided testing data.

        :param model: A trained model.
        :param test_data: A dictionary containing the testing features and target.
        """
        test_predictions = model.predict(test_data['RW11']['features']).flatten()

        mae = mean_absolute_error(test_data['RW11']['target'], test_predictions)
        mse = mean_squared_error(test_data['RW11']['target'], test_predictions)

        print(f'Mean Absolute Error: {mae:.2f}')
        print(f'Mean Squared Error: {mse:.2f}')
        print()

        # Plot the predictions
        a = plt.axes(aspect='equal')
        plt.scatter(test_data['RW11']['target'], test_predictions)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        lims = [0, 2]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)
        plt.show()