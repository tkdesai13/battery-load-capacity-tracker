import tensorflow as tf
from tcn import TCN
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
from typing import Dict, Any

class Model:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.model_dir = os.path.join(config['paths']['result'], config['experiment_name'])
        self.monitor_loss = self.config['model_config']['monitor']
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # Define paths for TensorBoard logs and model checkpoints
        self.tensorboard_log_dir = os.path.join(self.model_dir, 'logs')
        if not os.path.exists(self.tensorboard_log_dir):
            os.makedirs(self.tensorboard_log_dir)

        self.checkpoints_dir = os.path.join(self.model_dir, 'checkpoints')
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)

    def build_scalar_model(self, no_of_input_features: int):
        """
        Build and compile a neural network model with two inputs based on the provided configuration.
        """

        # Define input layers
        timebins_input = tf.keras.layers.Input(shape=(no_of_input_features,), name='timebins_input')
        previous_capacity_input = tf.keras.layers.Input(shape=(1,), name='previous_capacity_input')

        # Concatenate the inputs
        concatenated = tf.keras.layers.Concatenate()([timebins_input, previous_capacity_input])

        # Define the model architecture
        x = tf.keras.layers.Dense(64, activation='tanh')(concatenated)
        x = tf.keras.layers.Dense(32, activation='tanh')(x)
        output = tf.keras.layers.Dense(1, activation='linear')(x)

        # Build the model
        self.model = tf.keras.models.Model(inputs=[timebins_input, previous_capacity_input], outputs=output)

        # Compile the model
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])

    def build_vector_model(self, max_sequence_length: int, no_of_input_features: int):
        """
        Build and compile a 1D Convolutional Neural Network model with two inputs.
        The model assumes that the timebins input has already been padded to max_sequence_length.

        :param max_sequence_length: The maximum length of sequences after padding.
        :param no_of_input_features: The number of input features per time step.
        """

        # Define input layer for timebins with fixed length (after padding externally)
        timebins_input = tf.keras.layers.Input(shape=(max_sequence_length, no_of_input_features),
                                               name='timebins_input')

        # Mask the padded values (assuming padded values are 0)
        masked_timebins = tf.keras.layers.Masking(mask_value=999999.0)(timebins_input)

        # Apply 1D Convolution
        x = tf.keras.layers.Conv1D(filters=2, kernel_size=100, strides=4, activation='relu')(masked_timebins)
        x = tf.keras.layers.Conv1D(filters=4, kernel_size=50, strides=1, activation='relu')(x)

        # Apply max pooling
        # x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)

        # Flatten the output of the CNN to feed into dense layers
        x = tf.keras.layers.Flatten()(x)

        # Define the input layer for previous capacity (shape = (1,))
        previous_capacity_input = tf.keras.layers.Input(shape=(1,), name='previous_capacity_input')

        # Concatenate the flattened CNN output with the previous_capacity_input
        concatenated = tf.keras.layers.Concatenate()([x, previous_capacity_input])

        # Define dense layers after concatenation
        x = tf.keras.layers.Dense(64, activation='tanh')(concatenated)
        x = tf.keras.layers.Dense(32, activation='tanh')(x)
        output = tf.keras.layers.Dense(1, activation='linear')(x)

        # Build the model
        self.model = tf.keras.models.Model(inputs=[timebins_input, previous_capacity_input], outputs=output)

        # Save model summary
        model_summary_path = os.path.join(self.model_dir, 'model_summary.txt')
        with open(model_summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Compile the model
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss='mean_squared_error',
                           metrics=['mean_absolute_error', 'mean_squared_error'])

        return self.model

    def build_vector_model_TCN(self, max_sequence_length: int, no_of_input_features:int):
        """
        Build and compile a Temporal Convolutional Network (TCN) model with embeddings.
        """

        # Input layer for bin indices (integer-encoded sequence)
        bin_index_input = tf.keras.layers.Input(shape=(max_sequence_length,), name='bin_index_input')

        # Input layer for time spent in bins
        time_spent_input = tf.keras.layers.Input(shape=(max_sequence_length,), name='time_spent_input')

        # Embedding layer for bin indices
        embedding_dim = 16  # Adjust as needed
        input_dim = no_of_input_features+1  # 36 bins + 1 for padding (assuming bin indices from 1 to 36, padding index 0)
        bin_embedding = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim,
                                                  input_length=max_sequence_length, mask_zero=True)(bin_index_input)

        # Reshape time_spent_input to (batch_size, max_sequence_length, 1)
        time_spent_expanded = tf.keras.layers.Reshape((max_sequence_length, 1))(time_spent_input)

        # Apply masking to time_spent_input
        time_spent_masked = tf.keras.layers.Masking(mask_value=0.0)(time_spent_expanded)

        # Multiply bin_embedding by time_spent_masked to incorporate time spent
        scaled_embedding = tf.keras.layers.Multiply()([bin_embedding, time_spent_masked])

        # TCN layer
        x = TCN(nb_filters=8,  # Number of filters
                kernel_size=5,  # Kernel size
                dilations=[1, 2, 4, 8, 16, 32],  # Dilation rates
                activation='relu',
                padding='same',
                use_skip_connections=True)(scaled_embedding)

        # Input layer for previous capacity
        previous_capacity_input = tf.keras.layers.Input(shape=(1,), name='previous_capacity_input')

        # Concatenate TCN output with previous_capacity_input
        concatenated = tf.keras.layers.Concatenate()([x, previous_capacity_input])

        # Dense layers with regularization
        x = tf.keras.layers.Dense(64, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.00))(
            concatenated)
        x = tf.keras.layers.Dropout(0.0)(x)
        x = tf.keras.layers.Dense(32, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.00))(x)
        x = tf.keras.layers.Dropout(0.0)(x)
        output = tf.keras.layers.Dense(1, activation='linear')(x)

        # Build the model
        self.model = tf.keras.models.Model(
            inputs=[bin_index_input, time_spent_input, previous_capacity_input],
            outputs=output
        )

        # Save model summary
        model_summary_path = os.path.join(self.model_dir, 'model_summary.txt')
        with open(model_summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Compile the model
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_squared_error']
        )

        return self.model

    def train_model(self, train_data: Dict[str, Any], val_data: Dict[str, Any]):
        """
        Train the model using the provided training data.

        :param train_data: A dictionary containing the training features and target.
        :param val_data: A dictionary containing the validation features and target.
        """
        # Create callbacks for early stopping, TensorBoard logging, and model checkpoints
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor_loss,
            patience=self.config['model_specifications']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'model.h5'),
            monitor=self.monitor_loss,
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Train the model
        self.model.fit(
            [np.array(train_data['timebins']), np.array(train_data['previous_capacity'])],
            np.array(train_data['capacity_difference']),
            epochs=self.config['model_specifications']['epochs'],
            batch_size=self.config['model_specifications']['batch_size'],
            validation_data=(
                [np.array(val_data['timebins']), np.array(val_data['previous_capacity'])],
                np.array(val_data['capacity_difference'])
            ),
            callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback],
            verbose=1
        )

    def train_vector_model(self, train_data: Dict[str, Any], val_data: Dict[str, Any], max_sequence_length: int):
        """
        Train the model using the provided training data and apply padding to timebins sequences.

        :param train_data: A dictionary containing the training features and target.
        :param val_data: A dictionary containing the validation features and target.
        :param max_sequence_length: The maximum sequence length for padding.
        """

        # Pad the timebins sequences for both train and validation data
        padded_train_timebins = pad_sequences(train_data['timebins'], value=999999.0, maxlen=max_sequence_length, padding='pre',
                                              dtype='float32')
        padded_val_timebins = pad_sequences(val_data['timebins'], value=999999.0, maxlen=max_sequence_length, padding='pre',
                                            dtype='float32')


        # Create callbacks for early stopping, TensorBoard logging, and model checkpoints
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor_loss,
            patience=self.config['model_specifications']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'model.h5'),
            monitor=self.monitor_loss,
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Train the model
        self.model.fit(
            [tf.convert_to_tensor(padded_train_timebins), tf.convert_to_tensor(train_data['previous_capacity'])],
            tf.convert_to_tensor(train_data['capacity_difference']),
            epochs=self.config['model_specifications']['epochs'],
            batch_size=self.config['model_specifications']['batch_size'],
            validation_data=(
                [tf.convert_to_tensor(padded_val_timebins), tf.convert_to_tensor(val_data['previous_capacity'])],
                tf.convert_to_tensor(val_data['capacity_difference'])
            ),
            callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback],
            verbose=1
        )

    def train_vector_model_TCN(self, train_data: Dict[str, Any], val_data: Dict[str, Any], max_sequence_length: int):
        """
        Train the model using the provided training data and apply padding to sequences.

        :param train_data: A dictionary containing the training features and target.
        :param val_data: A dictionary containing the validation features and target.
        :param max_sequence_length: The maximum sequence length for padding.
        """

        # Pad the bin_indices sequences for both train and validation data
        padded_train_bin_indices = pad_sequences(
            train_data['bin_indices'],
            maxlen=max_sequence_length,
            padding='post',
            value=0  # Padding index for bin_indices
        )

        padded_val_bin_indices = pad_sequences(
            val_data['bin_indices'],
            maxlen=max_sequence_length,
            padding='post',
            value=0  # Padding index for bin_indices
        )

        # Pad the time_spent sequences for both train and validation data
        padded_train_time_spent = pad_sequences(
            train_data['timebins'],
            maxlen=max_sequence_length,
            padding='post',
            value=0.0  # Padding value for time_spent
        )

        padded_val_time_spent = pad_sequences(
            val_data['timebins'],
            maxlen=max_sequence_length,
            padding='post',
            value=0.0  # Padding value for time_spent
        )

        # Convert previous_capacity to numpy arrays and reshape to (batch_size, 1)
        train_previous_capacity = np.array(train_data['previous_capacity']).reshape(-1, 1)
        val_previous_capacity = np.array(val_data['previous_capacity']).reshape(-1, 1)

        # Convert capacity_difference to numpy arrays
        train_targets = np.array(train_data['capacity_difference'])
        val_targets = np.array(val_data['capacity_difference'])

        # Ensure data types are correct
        padded_train_bin_indices = padded_train_bin_indices.astype('int32')
        padded_val_bin_indices = padded_val_bin_indices.astype('int32')
        padded_train_time_spent = padded_train_time_spent.astype('float32')
        padded_val_time_spent = padded_val_time_spent.astype('float32')
        train_previous_capacity = train_previous_capacity.astype('float32')
        val_previous_capacity = val_previous_capacity.astype('float32')
        train_targets = train_targets.astype('float32')
        val_targets = val_targets.astype('float32')

        # Optionally, scale previous_capacity and time_spent
        # For example, using StandardScaler or MinMaxScaler

        # Create callbacks for early stopping, TensorBoard logging, and model checkpoints
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor=self.monitor_loss,
            patience=self.config['model_specifications']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'model.h5'),
            monitor=self.monitor_loss,
            save_best_only=True,
            mode='min',
            verbose=1
        )

        # Prepare the training and validation inputs as dictionaries
        train_inputs = {
            'bin_index_input': padded_train_bin_indices,
            'time_spent_input': padded_train_time_spent,
            'previous_capacity_input': train_previous_capacity
        }

        val_inputs = {
            'bin_index_input': padded_val_bin_indices,
            'time_spent_input': padded_val_time_spent,
            'previous_capacity_input': val_previous_capacity
        }

        # Train the model
        self.model.fit(
            x=train_inputs,
            y=train_targets,
            epochs=self.config['model_specifications']['epochs'],
            batch_size=self.config['model_specifications']['batch_size'],
            validation_data=(val_inputs, val_targets),
            callbacks=[early_stopping, tensorboard_callback, model_checkpoint_callback],
            verbose=1
        )

    def load_model(self, exp_path: str):
        """
        Load a saved model from the configured directory.
        """
        trained_model_path = os.path.join(exp_path, 'checkpoints', 'model.h5')
        self.model = tf.keras.models.load_model(trained_model_path)
