import os
import numpy as np
import random
import tensorflow as tf
import keras
# from tcn import TCN
from keras.utils import pad_sequences
# from keras.preprocessing.sequence import pad_sequences
# from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Conv1D, LSTM, Dense, Concatenate, TimeDistributed, Masking, Flatten, Dropout
from keras.models import Model
from typing import Dict, Any

# # Set random seeds for reproducibility
# seed=42
# np.random.seed(seed)
# random.seed(seed)
# tf.random.set_seed(seed)

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
        timebins_input = keras.layers.Input(shape=(no_of_input_features,), name='timebins_input')
        previous_capacity_input = keras.layers.Input(shape=(1,), name='previous_capacity_input')

        # Concatenate the inputs
        concatenated = keras.layers.Concatenate()([timebins_input, previous_capacity_input])

        # Define the model architecture
        x = keras.layers.Dense(64, activation='tanh')(concatenated)
        x = keras.layers.Dense(32, activation='tanh')(x)
        output = keras.layers.Dense(1, activation='linear')(x)

        # Build the model
        self.model = keras.models.Model(inputs=[timebins_input, previous_capacity_input], outputs=output)

        # Compile the model
        type_of_loss = self.config['model_config']['type_of_loss']
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=type_of_loss,
                           metrics=['mean_absolute_error', 'mean_squared_error'])

    def build_vector_model(self, max_sequence_length: int, no_of_input_features: int):
        """
        Build and compile a 1D Convolutional Neural Network model with two inputs.
        The model assumes that the timebins input has already been padded to max_sequence_length.

        :param max_sequence_length: The maximum length of sequences after padding.
        :param no_of_input_features: The number of input features per time step.
        """

        # Define input layer for timebins with fixed length (after padding externally)
        timebins_input = keras.layers.Input(shape=(max_sequence_length, no_of_input_features),
                                               name='timebins_input')

        # Mask the padded values (assuming padded values are 0.0)
        masked_timebins = keras.layers.Masking(mask_value=0.0)(timebins_input)

        # Adjusted TCN layer with increased receptive field and stacked layers
        # x = TCN(nb_filters=4,  # Number of filters
        #         kernel_size=20,  # Increased kernel size
        #         nb_stacks=2,  # Number of stacks (layers)
        #         dilations=[ 2, 4, 8, 16, 32, 64],  # Extended dilation rates
        #         activation='relu',  # Activation function
        #         padding='causal',  # Use causal padding
        #         use_skip_connections=True,
        #         dropout_rate=0.0,
        #         return_sequences=False)(masked_timebins)

        # Apply 1D Convolution - Layer 1
        x = keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh')(masked_timebins)
        x = keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh')(x)
        x = keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh')(x)
        x = keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh')(x)

        # Apply max pooling
        # x = keras.layers.MaxPooling1D(pool_size=2)(x)

        # Flatten the output of the CNN to feed into dense layers
        x = keras.layers.Flatten()(x)

        # Define the input layer for previous capacity (shape = (1,))
        previous_capacity_input = keras.layers.Input(shape=(1,), name='previous_capacity_input')

        # Concatenate the flattened CNN output with the previous_capacity_input
        concatenated = keras.layers.Concatenate()([x, previous_capacity_input])

        # Define dense layers after concatenation

        # Dense layers with regularization
        x = keras.layers.Dense(5, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.00))(
            concatenated)
        x = keras.layers.Dropout(0.0)(x)
        x = keras.layers.Dense(3, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.00))(x)
        x = keras.layers.Dropout(0.0)(x)
        output = keras.layers.Dense(1, activation='linear')(x)

        # Build the model
        self.model = keras.models.Model(inputs=[timebins_input, previous_capacity_input], outputs=output)

        # Save model summary
        model_summary_path = os.path.join(self.model_dir, 'model_summary.txt')
        with open(model_summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Compile the model
        type_of_loss = self.config['model_config']['type_of_loss']
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=type_of_loss,
                           metrics=['mean_absolute_error', 'mean_squared_error'])

        return self.model

    def build_vector_model_TCN(self, max_sequence_length: int, no_of_input_features:int):
        """
        Build and compile a Temporal Convolutional Network (TCN) model with embeddings.
        """

        # Input layer for bin indices (integer-encoded sequence)
        bin_index_input = keras.layers.Input(shape=(max_sequence_length,), name='bin_index_input')

        # Input layer for time spent in bins
        time_spent_input = keras.layers.Input(shape=(max_sequence_length,), name='time_spent_input')

        # Embedding layer for bin indices
        embedding_dim = self.config['model_config']['embedding_size']  # Adjust as needed
        input_dim = no_of_input_features+1  # 36 bins + 1 for padding (assuming bin indices from 1 to 36, padding index 0)
        bin_embedding = keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim,
                                                  input_length=max_sequence_length, mask_zero=True)(bin_index_input)

        # Reshape time_spent_input to (batch_size, max_sequence_length, 1)
        time_spent_expanded = keras.layers.Reshape((max_sequence_length, 1))(time_spent_input)

        # Apply masking to time_spent_input
        time_spent_masked = keras.layers.Masking(mask_value=0.0)(time_spent_expanded)

        # Interaction via element-wise multiplication
        interaction_features = keras.layers.Multiply()([bin_embedding, time_spent_masked])

        # Concatenate with original features
        combined_features = keras.layers.Concatenate()([bin_embedding, time_spent_masked, interaction_features])

        # # TCN layer
        # x = TCN(nb_filters=4,  # Number of filters
        #         kernel_size=10,  # Kernel size
        #         dilations=[1, 2, 4, 8, 16, 32,64],  # Dilation rates
        #         activation='relu',
        #         padding='same',
        #         use_skip_connections=True)(combined_features)

        # x = TCN(nb_filters=4,  # Number of filters
        #         kernel_size=20,  # Increased kernel size
        #         nb_stacks=1,  # Number of stacks (layers)
        #         dilations=[2, 4, 8, 16, 32, 64],  # Extended dilation rates
        #         activation='relu',  # Activation function
        #         padding='same',  # Use causal padding
        #         use_skip_connections=True,
        #         dropout_rate=0.0,
        #         return_sequences=False)(combined_features)

        # Input layer for previous capacity
        previous_capacity_input = keras.layers.Input(shape=(1,), name='previous_capacity_input')

        # Concatenate TCN output with previous_capacity_input
        concatenated = keras.layers.Concatenate()([x, previous_capacity_input])

        # Dense layers with regularization
        x = keras.layers.Dense(5, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.00))(
            concatenated)
        x = keras.layers.Dropout(0.0)(x)
        x = keras.layers.Dense(3, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.00))(x)
        x = keras.layers.Dropout(0.0)(x)
        output = keras.layers.Dense(1, activation='linear')(x)

        # Build the model
        self.model = keras.models.Model(
            inputs=[bin_index_input, time_spent_input, previous_capacity_input],
            outputs=output
        )

        # Save model summary
        model_summary_path = os.path.join(self.model_dir, 'model_summary.txt')
        with open(model_summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Compile the model
        type_of_loss = self.config['model_config']['type_of_loss']
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=type_of_loss,
            metrics=['mean_absolute_error', 'mean_squared_error']
        )

        return self.model

    def build_vector_model_embedded_convo(self, max_sequence_length: int, no_of_input_features:int):
        """
        Build and compile a 1D Convolution model with embeddings.
        """

        # Input layer for bin indices (integer-encoded sequence)
        bin_index_input = keras.layers.Input(shape=(max_sequence_length,), name='bin_index_input')

        # Input layer for time spent in bins
        time_spent_input = keras.layers.Input(shape=(max_sequence_length,), name='time_spent_input')

        # Embedding layer for bin indices
        embedding_dim = self.config['model_config']['embedding_size']  # Adjust as needed
        input_dim = no_of_input_features+1  # 36 bins + 1 for padding (assuming bin indices from 1 to 36, padding index 0)
        bin_embedding = keras.layers.Embedding(input_dim=input_dim, output_dim=embedding_dim,
                                                  input_length=max_sequence_length, mask_zero=True)(bin_index_input)

        # Reshape time_spent_input to (batch_size, max_sequence_length, 1)
        time_spent_expanded = keras.layers.Reshape((max_sequence_length, 1))(time_spent_input)

        # Apply masking to time_spent_input
        time_spent_masked = keras.layers.Masking(mask_value=0.0)(time_spent_expanded)

        # Interaction via element-wise multiplication
        interaction_features = keras.layers.Multiply()([bin_embedding, time_spent_masked])

        # Concatenate with original features
        combined_features = keras.layers.Concatenate(name='first_concatenate')([bin_embedding, time_spent_masked, interaction_features])

        # # TCN layer
        # x = TCN(nb_filters=4,  # Number of filters
        #         kernel_size=10,  # Kernel size
        #         dilations=[1, 2, 4, 8, 16, 32,64],  # Dilation rates
        #         activation='relu',
        #         padding='same',
        #         use_skip_connections=True)(combined_features)

        # Apply 1D Convolution - Layer 1
        x = keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Convo1')(combined_features)
        x = keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Convo2')(x)
        x = keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Convo3')(x)
        x = keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Convo4')(x)

        # Apply max pooling
        # x = keras.layers.MaxPooling1D(pool_size=2)(x)

        # Flatten the output of the CNN to feed into dense layers
        x = keras.layers.Flatten()(x)

        # x = TCN(nb_filters=4,  # Number of filters
        #         kernel_size=20,  # Increased kernel size
        #         nb_stacks=1,  # Number of stacks (layers)
        #         dilations=[2, 4, 8, 16, 32, 64],  # Extended dilation rates
        #         activation='relu',  # Activation function
        #         padding='same',  # Use causal padding
        #         use_skip_connections=True,
        #         dropout_rate=0.0,
        #         return_sequences=False)(combined_features)

        # Input layer for previous capacity
        previous_capacity_input = keras.layers.Input(shape=(1,), name='previous_capacity_input')

        # Concatenate TCN output with previous_capacity_input
        concatenated = keras.layers.Concatenate(name='second_concatenate')([x, previous_capacity_input])

        # Dense layers with regularization
        x = keras.layers.Dense(5, activation='tanh', name='first_dense', kernel_regularizer=keras.regularizers.l2(0.00))(
            concatenated)
        x = keras.layers.Dropout(0.0)(x)
        x = keras.layers.Dense(3, activation='tanh', name='second_dense', kernel_regularizer=keras.regularizers.l2(0.00))(x)
        x = keras.layers.Dropout(0.0)(x)
        output = keras.layers.Dense(1, name='third_dense', activation='linear')(x)

        # Build the model
        self.model = keras.models.Model(
            inputs=[bin_index_input, time_spent_input, previous_capacity_input],
            outputs=output
        )

        # Save model summary
        model_summary_path = os.path.join(self.model_dir, 'model_summary.txt')
        with open(model_summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))

        # Compile the model
        type_of_loss = self.config['model_config']['type_of_loss']
        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss=type_of_loss,
            metrics=['mean_absolute_error', 'mean_squared_error']
        )

        return self.model

    def build_sequence_to_sequence_model(self,
                                         encoder_timesteps: int,
                                         decoder_timesteps: int,
                                         transitions: int,
                                         bins: int,
                                         no_of_input_features: int):
        """
        Build and compile a Sequence-to-Sequence model with Encoder-Decoder architecture.

        :param encoder_timesteps: Number of time steps for the encoder (e.g., 10).
        :param decoder_timesteps: Number of time steps for the decoder (e.g., 15).
        :param transitions: Number of transitions in each U matrix.
        :param bins: Number of bins in each U matrix.
        :param no_of_input_features: Number of input features per U matrix (transitions * bins).
        """

        # ============================
        # 1. Encoder Inputs
        # ============================

        # Input for U matrices in encoder
        encoder_U_input = Input(shape=(encoder_timesteps, transitions * bins), name='encoder_U_input')

        # Input for deltaQ in encoder
        encoder_deltaQ_input = Input(shape=(encoder_timesteps, 1), name='encoder_deltaQ_input')

        # Input for initial Q
        initial_Q_input = Input(shape=(1,), name='initial_Q_input')

        # Masking for encoder U input
        encoder_U_masked = Masking(mask_value=0.0)(encoder_U_input)

        # ============================
        # 2. Encoder Convolutional Layers
        # ============================

        # Apply TimeDistributed Conv1D to process each U matrix
        x = TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))(encoder_U_masked)
        x = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))(x)
        x = TimeDistributed(Flatten())(x)  # Flatten to combine transitions and bins

        # Concatenate deltaQ inputs
        x = Concatenate(axis=-1)(
            [x, encoder_deltaQ_input])  # Shape: (batch_size, encoder_timesteps, 32*transitions*bins + 1)

        # Incorporate initial Q into each time step
        # Expand initial Q to match encoder_timesteps
        initial_Q_expanded = keras.layers.RepeatVector(encoder_timesteps)(initial_Q_input)
        x = Concatenate(axis=-1)([x, initial_Q_expanded])  # Shape: (batch_size, encoder_timesteps, ... + 1)

        # ============================
        # 3. Encoder Recurrent Layer
        # ============================

        encoder_lstm = LSTM(128, return_state=True, return_sequences=False)
        encoder_outputs, state_h, state_c = encoder_lstm(x)
        encoder_states = [state_h, state_c]

        # ============================
        # 4. Decoder Inputs
        # ============================

        # Input for U matrices in decoder
        decoder_U_input = Input(shape=(decoder_timesteps, transitions * bins), name='decoder_U_input')

        # Input for deltaQ in decoder (during training, use true deltaQ; during inference, use predicted deltaQ)
        decoder_deltaQ_input = Input(shape=(decoder_timesteps, 1), name='decoder_deltaQ_input')

        # Masking for decoder U input
        decoder_U_masked = Masking(mask_value=0.0)(decoder_U_input)

        # ============================
        # 5. Decoder Convolutional Layers
        # ============================

        # Apply TimeDistributed Conv1D to process each U matrix in decoder
        y = TimeDistributed(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))(decoder_U_masked)
        y = TimeDistributed(Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'))(y)
        y = TimeDistributed(Flatten())(y)  # Flatten to combine transitions and bins

        # Concatenate deltaQ inputs
        y = Concatenate(axis=-1)(
            [y, decoder_deltaQ_input])  # Shape: (batch_size, decoder_timesteps, 32*transitions*bins + 1)

        # ============================
        # 6. Decoder Recurrent Layer
        # ============================

        decoder_lstm = LSTM(128, return_sequences=True, return_state=False)
        decoder_outputs = decoder_lstm(y, initial_state=encoder_states)  # Use encoder states as initial states

        # ============================
        # 7. Output Layer
        # ============================

        # Predict deltaQ for each time step
        deltaQ_pred = TimeDistributed(Dense(1, activation='linear'), name='deltaQ_output')(decoder_outputs)

        # ============================
        # 8. Building the Model
        # ============================

        # Define the model with encoder and decoder inputs
        model = Model(inputs=[encoder_U_input, encoder_deltaQ_input, initial_Q_input,
                              decoder_U_input, decoder_deltaQ_input],
                      outputs=deltaQ_pred)

        # ============================
        # 9. Compile the Model
        # ============================

        learning_rate = self.config['model_specifications']['learning_rate']
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error', 'mean_squared_error'])

        # ============================
        # 10. Save Model Summary
        # ============================

        model_summary_path = os.path.join(self.model_dir, 'model_summary.txt')
        with open(model_summary_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        # ============================
        # 11. Return the Model
        # ============================

        self.model = model
        return self.model

    def train_model(self, train_data: Dict[str, Any], val_data: Dict[str, Any]):
        """
        Train the model using the provided training data.

        :param train_data: A dictionary containing the training features and target.
        :param val_data: A dictionary containing the validation features and target.
        """
        # Create callbacks for early stopping, TensorBoard logging, and model checkpoints
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=self.monitor_loss,
            patience=self.config['model_specifications']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'model.keras'),
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
        padded_train_timebins = pad_sequences(train_data['timebins'], value=0.0, maxlen=max_sequence_length, padding='pre',
                                              dtype='float32')
        padded_val_timebins = pad_sequences(val_data['timebins'], value=0.0, maxlen=max_sequence_length, padding='pre',
                                            dtype='float32')


        # Create callbacks for early stopping, TensorBoard logging, and model checkpoints
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=self.monitor_loss,
            patience=self.config['model_specifications']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'model.keras'),
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
            value=0.0, # Padding value for time_spent
            dtype='float32'
        )

        padded_val_time_spent = pad_sequences(
            val_data['timebins'],
            maxlen=max_sequence_length,
            padding='post',
            value=0.0,
            dtype='float32' # Padding value for time_spent
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
        early_stopping = keras.callbacks.EarlyStopping(
            monitor=self.monitor_loss,
            patience=self.config['model_specifications']['early_stopping_patience'],
            restore_best_weights=True
        )

        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=self.tensorboard_log_dir,
            histogram_freq=1
        )

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self.checkpoints_dir, 'model.keras'),
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
        trained_model_path = os.path.join(exp_path, 'checkpoints', 'model.keras')
        # self.model = keras.models.load_model(trained_model_path)
        self.model.load_weights(trained_model_path)