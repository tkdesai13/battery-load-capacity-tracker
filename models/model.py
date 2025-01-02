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

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')


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
        # interaction_features = keras.layers.Multiply()([bin_embedding, time_spent_masked])

        # Concatenate with original features
        combined_features = keras.layers.Concatenate(name='first_concatenate')([bin_embedding, time_spent_masked])

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

    def build_sequence_to_sequence_model_with_padding_vector_and_convolution(self,
                                                                             max_encoder_timesteps: int,
                                                                             max_decoder_timesteps: int,
                                                                             max_transitions: int,
                                                                             bins: int):
        """
        Build and compile an Encoder-Decoder Sequence-to-Sequence model using convolution-based feature extraction
        with masking to handle padding.

        :param max_encoder_timesteps: Maximum number of timesteps for the encoder.
        :param max_decoder_timesteps: Maximum number of timesteps for the decoder.
        :param max_transitions: Maximum number of transitions in the bin matrix.
        :param bins: Number of bins.
        """
        embedding_dim = self.config['model_config']['embedding_size']
        type_of_loss = self.config['model_config']['type_of_loss']
        learning_rate = self.config['model_specifications']['learning_rate']
        latent_state_neuron = self.config['model_config']['latent_state_neuron']

        # ============================
        # 1. Encoder Inputs
        # ============================

        # Input for bin indices (sequence of transitions for encoder)
        encoder_bin_indices = keras.layers.Input(
            shape=(max_encoder_timesteps, max_transitions), name="encoder_bin_indices"
        )

        # Input for time spent in bins (corresponding to encoder bin indices)
        encoder_time_spent = keras.layers.Input(
            shape=(max_encoder_timesteps, max_transitions), name="encoder_time_spent"
        )

        # Input for capacity delta
        encoder_capacity_delta = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_capacity_delta"
        )

        # Embedding for bin indices with masking

        timebin_encoding_layer = keras.layers.TimeDistributed(
            keras.layers.Embedding(
                input_dim=bins + 1,  # +1 for padding
                output_dim=embedding_dim,
                mask_zero=True,  # Enables masking for padded values
            ),
            name="encoder_embedding",
        )

        convoluting_encoded_data = keras.layers.TimeDistributed(
            keras.Sequential([
                keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Convo1'),
                keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Convo2'),
                keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Convo3'),
                keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Convo4'),
                keras.layers.Flatten(),
            ]),
            name="encoder_convolution",
        )


        encoder_bin_embedding = timebin_encoding_layer(encoder_bin_indices)

        # Reshape time spent input to match embedding dimensions
        encoder_time_spent_expanded = keras.layers.Reshape(
            target_shape=(max_encoder_timesteps, max_transitions, 1)
        )(encoder_time_spent)

        # Apply masking to time spent
        # encoder_time_spent_masked = keras.layers.Masking(mask_value=0.0)(encoder_time_spent_expanded)

        # Interaction via element-wise multiplication
        encoder_interaction = keras.layers.Concatenate()(
            [encoder_bin_embedding, encoder_time_spent_expanded]
        )

        # # Process each timestep with Convolutional Layers
        # encoder_features = keras.layers.TimeDistributed(
        #     keras.Sequential([
        #         keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Convo1'),
        #         keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Convo2'),
        #         keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Convo3'),
        #         # keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Convo4'),
        #         keras.layers.Flatten(),
        #     ]),
        #     name="encoder_convolution",
        # )(encoder_interaction)

        encoder_features = convoluting_encoded_data(encoder_interaction)

        # Concatenate flattened features with encoder_capacity_delta
        encoder_combined_features = keras.layers.Concatenate(name="encoder_feature_concat")(
            [encoder_features, encoder_capacity_delta]
        )

        # ============================
        # 2. Encoder LSTM Layer with Masking
        # ============================

        encoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron, return_sequences=False, return_state=True, name="encoder_lstm"
        )
        encoder_output, state_h, state_c = encoder_lstm(encoder_combined_features)

        # ============================
        # 3. Decoder Inputs
        # ============================

        # Input for decoder bin indices (sequence of transitions for decoder)
        decoder_bin_indices = keras.layers.Input(
            shape=(max_decoder_timesteps, max_transitions), name="decoder_bin_indices"
        )

        # Input for time spent in bins (corresponding to decoder bin indices)
        decoder_time_spent = keras.layers.Input(
            shape=(max_decoder_timesteps, max_transitions), name="decoder_time_spent"
        )

        # # Embedding for decoder bin indices with masking
        # decoder_bin_embedding = keras.layers.TimeDistributed(
        #     keras.layers.Embedding(
        #         input_dim=bins + 1,  # +1 for padding
        #         output_dim=embedding_dim,
        #         mask_zero=True,  # Enables masking for padded values
        #     ),
        #     name="decoder_embedding",
        # )(decoder_bin_indices)

        decoder_bin_embedding = timebin_encoding_layer(decoder_bin_indices)

        # Reshape time spent input to match embedding dimensions
        decoder_time_spent_expanded = keras.layers.Reshape(
            target_shape=(max_decoder_timesteps, max_transitions, 1)
        )(decoder_time_spent)

        # Apply masking to time spent
        # decoder_time_spent_masked = keras.layers.Masking(mask_value=0.0)(decoder_time_spent_expanded)

        # Interaction via element-wise multiplication
        decoder_interaction = keras.layers.Concatenate()(
            [decoder_bin_embedding, decoder_time_spent_expanded]
        )

        # # Process each timestep with Convolutional Layers
        decoder_features = keras.layers.TimeDistributed(
            keras.Sequential([
                keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Decoder_Convo1'),
                keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Decoder_Convo2'),
                keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Decoder_Convo3'),
                # keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Decoder_Convo4'),
                keras.layers.Flatten(),
            ]),
            name="decoder_convolution",
        )(decoder_interaction)

        # decoder_features = convoluting_encoded_data(decoder_interaction)

        # ============================
        # 4. Decoder LSTM Layer with Initial States
        # ============================

        decoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron, return_sequences=True, return_state=False, name="decoder_lstm"
        )
        decoder_outputs = decoder_lstm(
            decoder_features, initial_state=[state_h, state_c]
        )

        # ============================
        # 5. Output Layer
        # ============================

        deltaQ_pred = keras.layers.TimeDistributed(
            keras.layers.Dense(1, activation="linear"), name="deltaQ_output"
        )(decoder_outputs)

        # ============================
        # 6. Build and Compile the Model
        # ============================

        model = keras.models.Model(
            inputs=[
                encoder_bin_indices,
                encoder_time_spent,
                encoder_capacity_delta,
                decoder_bin_indices,
                decoder_time_spent,
            ],
            outputs=deltaQ_pred,
        )
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=type_of_loss,
            metrics=["mean_absolute_error", "mean_squared_error"],
        )

        # Save model summary
        model_summary_path = os.path.join(self.model_dir, "model_summary.txt")
        with open(model_summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

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

    def pad_transitions(self, sequences, max_transitions):
        """
        Pad or truncate transitions at each timestep to ensure each timestep has exactly max_transitions entries.
        sequences: List of samples (N), each is a list of timesteps (T), each timestep is a list/array of variable length transitions.
        Returns: A new list of samples with each timestep padded to max_transitions.
        """
        padded = []
        for sample in sequences:
            padded_sample = []
            for timestep_data in sample:
                if len(timestep_data) < max_transitions:
                    # Pad with zeros at the end
                    padded_timestep = np.pad(timestep_data, (0, max_transitions - len(timestep_data)), 'constant',
                                             constant_values=0)
                else:
                    # Truncate if longer than max_transitions
                    padded_timestep = timestep_data[:max_transitions]
                padded_sample.append(padded_timestep)
            padded.append(np.array(padded_sample, dtype='float32'))
        return padded

    def pad_and_mask_data(self, data_split, max_encoder_timesteps, max_decoder_timesteps, max_transitions,
                          pre_pad_encoder=False):
        """
        Pad transitions to max_transitions, then pad encoder and decoder sequences to max_encoder_timesteps and max_decoder_timesteps respectively.

        data_split: dictionary with keys 'encoder_inputs', 'decoder_inputs', 'decoder_outputs'.
        Returns: (inputs, targets) tuple of padded arrays ready for model input.
        """
        # Encoder
        enc_bin_indices = self.pad_transitions(data_split['encoder_inputs']['bin_indices'], max_transitions)
        enc_time_spent = self.pad_transitions(data_split['encoder_inputs']['timebins'], max_transitions)
        # enc_capacity_diff = self.pad_transitions(data_split['encoder_inputs']['capacity_difference'], max_transitions)

        # Decoder
        dec_bin_indices = self.pad_transitions(data_split['decoder_inputs']['bin_indices'], max_transitions)
        dec_time_spent = self.pad_transitions(data_split['decoder_inputs']['timebins'], max_transitions)
        # dec_targets = self.pad_transitions(data_split['decoder_outputs']['capacity_difference'], max_transitions)

        # Now pad along the timesteps dimension
        enc_padding_type = 'pre'

        enc_bin_indices_padded = pad_sequences(enc_bin_indices, maxlen=max_encoder_timesteps, padding=enc_padding_type,
                                               value=0, dtype='int16')
        enc_time_spent_padded = pad_sequences(enc_time_spent, maxlen=max_encoder_timesteps, padding=enc_padding_type,
                                              value=0.0, dtype='float16')
        enc_capacity_diff_padded = pad_sequences(data_split['encoder_inputs']['capacity_difference'],
                                                 maxlen=max_encoder_timesteps,padding=enc_padding_type,
                                                 value=0.0, dtype='float16')
        # reshape encoder_capacity_diff_padded to (batch_size, max_encoder_timesteps, 1)
        enc_capacity_diff_padded = np.expand_dims(enc_capacity_diff_padded, axis=-1)
        enc_capacity_diff_padded = enc_capacity_diff_padded.astype('float16')


        dec_padding_type = 'post'

        dec_bin_indices_padded = pad_sequences(dec_bin_indices, maxlen=max_decoder_timesteps,
                                               padding=dec_padding_type, value=0, dtype='int16')
        dec_time_spent_padded = pad_sequences(dec_time_spent, maxlen=max_decoder_timesteps,
                                              padding=dec_padding_type, value=0.0, dtype='float16')
        dec_targets_padded = pad_sequences(data_split['decoder_outputs']['capacity_difference'],
                                           maxlen=max_decoder_timesteps, padding=dec_padding_type,
                                           value=0.0, dtype='float16')

        # Reshape targets to (batch_size, max_decoder_timesteps, 1)
        dec_targets_padded = np.expand_dims(dec_targets_padded, axis=-1)
        dec_targets_padded = dec_targets_padded.astype('float16')

        # Convert to appropriate dtypes
        # enc_bin_indices_padded = enc_bin_indices_padded.astype('int32')
        # enc_time_spent_padded = enc_time_spent_padded.astype('float32')
        # enc_capacity_diff_padded = enc_capacity_diff_padded.astype('float32')
        # dec_bin_indices_padded = dec_bin_indices_padded.astype('int32')
        # dec_time_spent_padded = dec_time_spent_padded.astype('float32')
        # dec_targets_padded = dec_targets_padded.astype('float32')

        inputs = {
            'encoder_bin_indices': tf.convert_to_tensor(enc_bin_indices_padded),
            'encoder_time_spent': tf.convert_to_tensor(enc_time_spent_padded),
            'encoder_capacity_delta': tf.convert_to_tensor(enc_capacity_diff_padded),
            'decoder_bin_indices': tf.convert_to_tensor(dec_bin_indices_padded),
            'decoder_time_spent': tf.convert_to_tensor(dec_time_spent_padded)
        }

        targets = tf.convert_to_tensor(dec_targets_padded)
        return inputs, targets

    def train_sequence_to_sequence_model_with_padding_vector_and_convolution(self, data: Dict[str, Any],max_encoder_timesteps: int, max_decoder_timesteps: int, max_transitions: int):
        """
        Train the sequence-to-sequence model using provided training and validation data with two-level padding and masking.
        """

        # Preprocess and pad training and validation data
        train_inputs, train_targets = self.pad_and_mask_data(data['train'], max_encoder_timesteps, max_decoder_timesteps, max_transitions, pre_pad_encoder=False)
        val_inputs, val_targets = self.pad_and_mask_data(data['validation'], max_encoder_timesteps, max_decoder_timesteps, max_transitions, pre_pad_encoder=False)

        # Callbacks
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