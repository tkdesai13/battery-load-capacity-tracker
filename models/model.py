import tensorflow as tf
import os
import numpy as np
from typing import Dict, Any


class Model:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_dir = os.path.join(config['paths']['result'], config['experiment_name'])
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Define paths for TensorBoard logs and model checkpoints
        self.tensorboard_log_dir = os.path.join(self.model_dir, 'logs')
        if not os.path.exists(self.tensorboard_log_dir):
            os.makedirs(self.tensorboard_log_dir)

        self.checkpoints_dir = os.path.join(self.model_dir, 'checkpoints')
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

    def build_model(self):
        """
        Build and compile a sequential neural network model based on the provided configuration.
        """
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.config['data']['num_features'],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])

    def train_model(self, train_data: Dict[str, Any], val_data: Dict[str, Any]):
        """
        Train the model using the provided training data.

        :param train_data: A dictionary containing the training features and target.
        :param val_data: A dictionary containing the validation features and target.
        """
        # Create callbacks for early stopping, TensorBoard logging, and model checkpoints
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config['training']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Train the model
        self.model.fit(
            np.array(train_data['features']), np.array(train_data['target']),
            epochs=self.config['training']['epochs'],
            validation_data=(np.array(val_data['features']), np.array(val_data['target'])),
            callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback],
            verbose=1
        )

    def load_model(self):
        """
        Load a saved model from the configured directory.
        """
        self.model = tf.keras.models.load_model(os.path.join(self.checkpoints_dir, 'best_model.h5'))

    def predict(self, data: Any):
        """
        Make predictions using the trained model.

        :param data: The input data for which to make predictions.
        :return: The model's predictions.
        """
        return self.model.predict(np.array(data))

    def evaluate(self, test_data: Dict[str, Any]):
        """
        Evaluate the model on the test data.

        :param test_data: A dictionary containing the testing features and target.
        :return: A tuple containing the loss, mean absolute error, and mean squared error.
        """
        loss, mae, mse = self.model.evaluate(np.array(test_data['features']), np.array(test_data['target']), verbose=0)
        return loss, mae, mse

    def run(self, train_data: Dict[str, Any], val_data: Dict[str, Any], test_data: Dict[str, Any]):
        """
        Build, train, save, and evaluate the model.

        :param train_data: A dictionary containing the training features and target.
        :param val_data: A dictionary containing the validation features and target.
        :param test_data: A dictionary containing the testing features and target.
        :return: A tuple containing the loss, mean absolute error, and mean squared error.
        """
        # Build the model
        self.build_model()

        # Train the model
        self.train_model(train_data, val_data)

        # Evaluate the model
        return self.evaluate(test_data)
