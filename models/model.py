# Description: This file contains class for building the model and training them.

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

    def build_model(self, input_features_size: int):
        """
        Build and compile a sequential neural network model based on the provided configuration.
        """

        # Define the model architecture
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='tanh', input_shape=(input_features_size,)),
            tf.keras.layers.Dense(32, activation='tanh'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        # Compile the model
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
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
            patience=self.config['model_specifications']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Train the model
        self.model.fit(
            np.array(train_data['features']), np.array(train_data['target']),
            epochs=self.config['model_specifications']['epochs'],
            validation_data=(np.array(val_data['features']), np.array(val_data['target'])),
            callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback],
            verbose=1
        )

    def load_model(self, exp_path: str):
        """
        Load a saved model from the configured directory.
        """
        trained_model_path = os.path.join(exp_path, 'checkpoints', 'model.h5')
        self.model = tf.keras.models.load_model(trained_model_path)

