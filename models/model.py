import os
import time
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
# Set global precision policy to float32
# policy = mixed_precision.Policy('float32')
# mixed_precision.set_global_policy(policy)

import logging

tf.get_logger().setLevel(logging.ERROR)


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
        self.embedding_layer = None
        self.encoder_model = None
        self.decoder_model = None
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # self.setup_mixed_precision()
        self.setup_optimizer_and_loss()
        self.setup_metrics()


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

    def build_sequence_to_sequence_model_with_table_or_cube_data(self,
                                                                 max_encoder_timesteps: int,
                                                                 max_decoder_timesteps: int,
                                                                 bins: int):
        """
        Build and compile an Encoder-Decoder Sequence-to-Sequence model using convolution-based feature extraction
        with masking to handle padding.

        :param max_encoder_timesteps: Maximum number of timesteps for the encoder.
        :param max_decoder_timesteps: Maximum number of timesteps for the decoder.
        :param max_transitions: Maximum number of transitions in the bin matrix.
        :param bins: Number of bins.
        """

        type_of_loss = self.config['model_config']['type_of_loss']
        learning_rate = self.config['model_specifications']['learning_rate']
        latent_state_neuron = self.config['model_config']['latent_state_neuron']

        # ============================
        # 1. Encoder Inputs
        # ============================

        # Input for time spent in bins (corresponding to encoder bin indices)
        encoder_time_spent = keras.layers.Input(
            shape=(max_encoder_timesteps, bins), name="encoder_time_spent"
        )

        # Input for capacity delta
        encoder_capacity_delta = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_capacity_delta"
        )

        # Input for cumulative time spent in bins on the encoder side
        encoder_cumulative_time_spent = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_cumulative_time_spent")

        # Input for encoder cumulative Ah throughput
        encoder_cumulative_Ah_throughput = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_cumulative_Ah_throughput")

        # Concatenate flattened features with encoder_capacity_delta
        encoder_combined_features = keras.layers.Concatenate(name="encoder_feature_concat")(
            [encoder_time_spent, encoder_capacity_delta, encoder_cumulative_time_spent, encoder_cumulative_Ah_throughput]
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

        # Input for time spent in bins (corresponding to decoder bin indices)
        decoder_time_spent = keras.layers.Input(
            shape=(max_decoder_timesteps, bins), name="decoder_time_spent")

        # Input for cumulative time spent in bins on the decoder side
        decoder_cumulative_time_spent = keras.layers.Input(
            shape=(max_decoder_timesteps, 1), name="decoder_cumulative_time_spent")

        # Input for decoder cumulative Ah throughput
        decoder_cumulative_Ah_throughput = keras.layers.Input(
            shape=(max_decoder_timesteps, 1), name="decoder_cumulative_Ah_throughput")

        # Concatenate flattened features with encoder_capacity_delta
        decoder_combined_features = keras.layers.Concatenate(name="decoder_feature_concat")(
            [decoder_time_spent, decoder_cumulative_time_spent, decoder_cumulative_Ah_throughput]
        )

        # ============================
        # 4. Decoder LSTM Layer with Initial States
        # ============================

        decoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron, return_sequences=True, return_state=False, name="decoder_lstm"
        )
        decoder_outputs = decoder_lstm(
            decoder_combined_features, initial_state=[state_h, state_c]
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
                encoder_time_spent,
                encoder_capacity_delta,
                encoder_cumulative_time_spent,
                encoder_cumulative_Ah_throughput,
                decoder_time_spent,
                decoder_cumulative_time_spent,
                decoder_cumulative_Ah_throughput
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


    def build_sequence_to_sequence_model_with_matrix_data(self,
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

        # helper: one Conv–Norm–ReLU–Pool block
        def conv_block(filters, k_size, stride, pool, name):
            return keras.Sequential([
                keras.layers.Conv1D(filters=filters,
                                    kernel_size=k_size,
                                    strides=stride,
                                    use_bias=False,
                                    kernel_initializer="he_normal",
                                    name=f"{name}_conv"),
                # keras.layers.LayerNormalization(name=f"{name}_ln"),
                keras.layers.Activation("relu", name=f"{name}_relu"),
                # keras.layers.MaxPooling1D(pool_size=pool,
                #                           padding="same",
                #                           name=f"{name}_pool"),
                keras.layers.Dropout(0.1, name=f"{name}_drop")  # ← optional, helps regularise
            ], name=f"{name}_block")



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

        # Input for encoder cumulative time spent
        encoder_cumulative_time_spent = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_cumulative_time_spent"
        )

        # Input for encoder cumulative Ah throughput
        encoder_cumulative_Ah_throughput = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_cumulative_Ah_throughput"
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

        # encoder CNN applied at every timestep
        encoder_features = keras.layers.TimeDistributed(
            keras.Sequential([
                conv_block(4, 15, 2, 4, name="enc_b1"),  # wider filters: 16 → 32 → 64 → 128 is common
                conv_block(8, 25, 2, 4, name="enc_b2"),
                conv_block(8, 35, 2, 2, name="enc_b3"),
                conv_block(16, 12, 2, 2, name="enc_b4"),
                keras.layers.Flatten(name="enc_flat")
            ], name="encoder_cnn"),
            name="encoder_convolution"
        )(encoder_interaction)

        # # Process each timestep with Convolutional Layers
        # encoder_features = keras.layers.TimeDistributed(
        #     keras.Sequential([
        #         keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='relu', name='Encoder_Convo1'),
        #         keras.layers.MaxPooling1D(pool_size=8, name='Encoder_pooling1'),
        #         keras.layers.Conv1D(filters=3, kernel_size=25, strides=4, activation='relu', name='Encoder_Convo2'),
        #         keras.layers.MaxPooling1D(pool_size=4, name='Encoder_pooling2'),
        #         keras.layers.Conv1D(filters=4, kernel_size=35, strides=4, activation='relu', name='Encoder_Convo3'),
        #         keras.layers.MaxPooling1D(pool_size=2, name='Encoder_pooling3'),
        #         keras.layers.Conv1D(filters=5, kernel_size=45, strides=5, activation='relu', name='Encoder_Convo4'),
        #         keras.layers.MaxPooling1D(pool_size=2, name='Encoder_pooling4'),
        #         keras.layers.Flatten(),
        #     ]),
        #     name="encoder_convolution",
        # )(encoder_interaction)

        # encoder_features = convoluting_encoded_data(encoder_interaction)

        # Concatenate flattened features with encoder_capacity_delta
        encoder_combined_features = keras.layers.Concatenate(name="encoder_feature_concat")(
            [encoder_features, encoder_capacity_delta, encoder_cumulative_time_spent, encoder_cumulative_Ah_throughput]
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

        # Input for decoder cumulative time spent
        decoder_cumulative_time_spent = keras.layers.Input(
            shape=(max_decoder_timesteps, 1), name="decoder_cumulative_time_spent"
        )

        # Input for decoder cumulative Ah throughput
        decoder_cumulative_Ah_throughput = keras.layers.Input(
            shape=(max_decoder_timesteps, 1), name="decoder_cumulative_Ah_throughput"
        )

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
        # decoder_features = keras.layers.TimeDistributed(
        #     keras.Sequential([
        #         keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='relu', name='Decoder_Convo1'),
        #         keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='relu', name='Decoder_Convo2'),
        #         keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='relu', name='Decoder_Convo3'),
        #         keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='relu', name='Decoder_Convo4'),
        #         keras.layers.Flatten(),
        #     ]),
        #     name="decoder_convolution",
        # )(decoder_interaction)

        decoder_conv_stack = keras.Sequential([
            conv_block(4, 15, 2, 4, name="dec_b1"),  # wider filters: 16 → 32 → 64 → 128 is common
            conv_block(8, 25, 2, 4, name="dec_b2"),
            conv_block(8, 35, 2, 2, name="dec_b3"),
            conv_block(16, 12, 2, 2, name="dec_b4"),
            keras.layers.Flatten(name="Dec_flat")
        ], name="decoder_conv_stack")

        decoder_features = keras.layers.TimeDistributed(
            decoder_conv_stack, name="decoder_convolution"
        )(decoder_interaction)  # shape → (batch, dec_steps, 128)


        # ============================
        # 4. Decoder LSTM Layer with Initial States
        # ============================

        decoderr_combined_features = keras.layers.Concatenate(name="decoder_feature_concat")(
            [decoder_features, decoder_cumulative_time_spent, decoder_cumulative_Ah_throughput]
        )

        decoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron, return_sequences=True, return_state=False, name="decoder_lstm"
        )
        decoder_outputs = decoder_lstm(
            decoderr_combined_features, initial_state=[state_h, state_c]
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
                encoder_cumulative_time_spent,
                encoder_cumulative_Ah_throughput,
                decoder_bin_indices,
                decoder_time_spent,
                decoder_cumulative_time_spent,
                decoder_cumulative_Ah_throughput
            ],
            outputs=deltaQ_pred,
        )
        # optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=type_of_loss,
            metrics=["mean_absolute_error", "mean_squared_error"],
        )

        # Print model summary
        print(model.summary())

        # Save model summary
        model_summary_path = os.path.join(self.model_dir, "model_summary.txt")
        with open(model_summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        self.model = model
        return self.model



    def build_sequence_to_sequence_model_with_previous_dq(self,
                                                          max_encoder_timesteps: int,
                                                          max_decoder_timesteps: int,
                                                          max_transitions: int,
                                                          bins: int):
        """
        Build and compile an Encoder-Decoder Sequence-to-Sequence model using convolution-based feature extraction
        with masking to handle padding, incorporating previous Delta Q predictions into the decoder.

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
            name="timebins_embedding",
        )

        # Embedding for bin indices with masking
        encoder_embedding = timebin_encoding_layer(encoder_bin_indices)

        # Reshape time spent input to match embedding dimensions
        encoder_time_spent_expanded = keras.layers.Reshape(
            target_shape=(max_encoder_timesteps, max_transitions, 1)
        )(encoder_time_spent)

        # Concatenate bin embeddings and time spent
        encoder_interaction = keras.layers.Concatenate()(
            [encoder_embedding, encoder_time_spent_expanded]
        )

        # Convolutional Layers for Encoder
        encoder_features = keras.layers.TimeDistributed(
            keras.Sequential([
                keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Convo1'),
                keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Convo2'),
                keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Convo3'),
                keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Convo4'),
                keras.layers.Flatten(),
            ]),
            name="encoder_convolution",
        )(encoder_interaction)

        # Concatenate flattened features with encoder_capacity_delta
        encoder_combined_features = keras.layers.Concatenate(name="encoder_feature_concat")(
            [encoder_features, encoder_capacity_delta]
        )

        # ============================
        # 2. Encoder LSTM Layer
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

        # Input for previous Delta Q predictions (ΔQ_{n-1} + Q(initial))
        decoder_previous_Q = keras.layers.Input(
            shape=(max_decoder_timesteps, 1), name="decoder_previous_Q"
        )

        # Embedding for decoder bin indices with masking
        decoder_embedding = timebin_encoding_layer(decoder_bin_indices)

        # Reshape time spent input to match embedding dimensions
        decoder_time_spent_expanded = keras.layers.Reshape(
            target_shape=(max_decoder_timesteps, max_transitions, 1)
        )(decoder_time_spent)

        # Concatenate decoder inputs with previous Delta Q
        decoder_interaction = keras.layers.Concatenate()(
            [decoder_embedding, decoder_time_spent_expanded]
        )

        # Convolutional Layers for Decoder
        decoder_features = keras.layers.TimeDistributed(
            keras.Sequential([
                keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Decoder_Convo1'),
                keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Decoder_Convo2'),
                keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Decoder_Convo3'),
                keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Decoder_Convo4'),
                keras.layers.Flatten(),
            ]),
            name="decoder_convolution",
        )(decoder_interaction)

        decoder_features_and_previous_cap = keras.layers.Concatenate()([decoder_features, decoder_previous_Q])

        # ============================
        # 4. Decoder LSTM Layer with Initial States
        # ============================

        decoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron, return_sequences=True, return_state=False, name="decoder_lstm"
        )
        decoder_outputs = decoder_lstm(
            decoder_features_and_previous_cap, initial_state=[state_h, state_c]
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
                decoder_previous_Q,
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
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_summary_path = os.path.join(self.model_dir, "model_summary.txt")
        with open(model_summary_path, "w") as f:
            model.summary(print_fn=lambda x: f.write(x + "\n"))

        self.model = model
        return self.model

    def create_shared_embedding(self, bins: int):
        """
        Create a shared embedding layer for both encoder and decoder.

        :param bins: Number of bins.
        :return: Shared Keras Embedding layer.
        """
        embedding_dim = self.config['model_config']['embedding_size']
        shared_embedding = keras.layers.Embedding(
            input_dim=bins + 1,  # +1 for padding
            output_dim=embedding_dim,
            mask_zero=True,  # Enables masking for padded values
            name="shared_embedding",
            dtype="float32"
        )
        return shared_embedding

    def build_encoder(self, max_encoder_timesteps: int, max_transitions: int, bins: int, embedding_layer):
        """
        Build the encoder part of the Seq2Seq model using a shared embedding layer.
        """
        embedding_dim = self.config['model_config']['embedding_size']
        latent_state_neuron = self.config['model_config']['latent_state_neuron']

        # Encoder Inputs with explicit dtypes
        encoder_bin_indices = keras.layers.Input(
            shape=(max_encoder_timesteps, max_transitions),
            name="encoder_bin_indices",
            dtype='int32'
        )

        encoder_time_spent = keras.layers.Input(
            shape=(max_encoder_timesteps, max_transitions),
            name="encoder_time_spent",
            dtype='float32'  # Explicitly set to float32
        )

        encoder_capacity_delta = keras.layers.Input(
            shape=(max_encoder_timesteps, 1),
            name="encoder_capacity_delta",
            dtype='float32'  # Explicitly set to float32
        )

        # Use the shared embedding layer
        encoder_embedding = embedding_layer(encoder_bin_indices)  # Shared embedding

        # Reshape time spent input to match embedding dimensions
        encoder_time_spent_expanded = keras.layers.Reshape(
            target_shape=(max_encoder_timesteps, max_transitions, 1)
        )(encoder_time_spent)

        # Concatenate bin embeddings and time spent
        encoder_interaction = keras.layers.Concatenate()(
            [encoder_embedding, encoder_time_spent_expanded]
        )

        # Convolutional Layers for Encoder
        encoder_features = keras.layers.TimeDistributed(
            keras.Sequential([
                keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Convo1'),
                keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Convo2'),
                keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Convo3'),
                keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Convo4'),
                keras.layers.Flatten(),
            ]),
            name="encoder_convolution",
        )(encoder_interaction)

        # Concatenate flattened features with encoder_capacity_delta
        encoder_combined_features = keras.layers.Concatenate(name="encoder_feature_concat")(
            [encoder_features, encoder_capacity_delta]
        )

        # Encoder LSTM Layer
        encoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron,
            return_sequences=False,
            return_state=True,
            name="encoder_lstm",
            dtype="float32"

        )
        encoder_output, state_h, state_c = encoder_lstm(encoder_combined_features)

        # Define the encoder model
        encoder_model = keras.models.Model(
            inputs=[
                encoder_bin_indices,
                encoder_time_spent,
                encoder_capacity_delta,
            ],
            outputs=[state_h, state_c],
            name="encoder_model"
        )

        return encoder_model

    # def build_decoder(self, max_decoder_timesteps: int, max_transitions: int, bins: int, embedding_layer):
    #     """
    #     Build the decoder part of the Seq2Seq model using a shared embedding layer.
    #     """
    #     embedding_dim = self.config['model_config']['embedding_size']
    #     latent_state_neuron = self.config['model_config']['latent_state_neuron']
    #
    #     # Decoder Inputs
    #     decoder_bin_indices = keras.layers.Input(
    #         shape=(max_decoder_timesteps, max_transitions), name="decoder_bin_indices", dtype='int32'
    #     )
    #
    #     decoder_time_spent = keras.layers.Input(
    #         shape=(max_decoder_timesteps, max_transitions), name="decoder_time_spent", dtype="float32"
    #     )
    #
    #     decoder_previous_Q = keras.layers.Input(
    #         shape=(max_decoder_timesteps, 1), name="decoder_previous_Q", dtype="float32"
    #     )
    #
    #     # Decoder Initial States
    #     decoder_state_h = keras.layers.Input(shape=(latent_state_neuron,), name="decoder_state_h", dtype="float32")
    #     decoder_state_c = keras.layers.Input(shape=(latent_state_neuron,), name="decoder_state_c", dtype="float32")
    #
    #     # Use the shared embedding layer
    #     decoder_embedding = embedding_layer(decoder_bin_indices)  # Shared embedding
    #
    #     # Reshape time spent input to match embedding dimensions
    #     decoder_time_spent_expanded = keras.layers.Reshape(
    #         target_shape=(max_decoder_timesteps, max_transitions, 1)
    #     )(decoder_time_spent)
    #
    #     # Concatenate decoder inputs (excluding previous_Q for now)
    #     decoder_interaction = keras.layers.Concatenate()(
    #         [decoder_embedding, decoder_time_spent_expanded]
    #     )
    #
    #     # Convolutional Layers for Decoder
    #     decoder_features = keras.layers.TimeDistributed(
    #         keras.Sequential([
    #             keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Decoder_Convo1'),
    #             keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Decoder_Convo2'),
    #             keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Decoder_Convo3'),
    #             keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Decoder_Convo4'),
    #             keras.layers.Flatten(),
    #         ]),
    #         name="decoder_convolution",
    #     )(decoder_interaction)
    #
    #     # Concatenate decoder_features with decoder_previous_Q
    #     decoder_features_and_previous_cap = keras.layers.Concatenate()(
    #         [decoder_features, decoder_previous_Q]
    #     )
    #
    #     # Decoder LSTM Layer with initial states
    #     decoder_lstm = keras.layers.LSTM(
    #         units=latent_state_neuron, return_sequences=True, return_state=True, name="decoder_lstm", dtype="float32"
    #     )
    #     decoder_outputs, new_state_h, new_state_c = decoder_lstm(
    #         decoder_features_and_previous_cap, initial_state=[decoder_state_h, decoder_state_c]
    #     )
    #
    #     # Output Layer
    #     deltaQ_pred = keras.layers.TimeDistributed(
    #         keras.layers.Dense(1, activation="linear", dtype="float32"), name="deltaQ_output"
    #     )(decoder_outputs)
    #
    #     # Define the decoder model with outputs including new states
    #     decoder_model = keras.models.Model(
    #         inputs=[
    #             decoder_bin_indices,
    #             decoder_time_spent,
    #             decoder_previous_Q,
    #             decoder_state_h,
    #             decoder_state_c
    #         ],
    #         outputs=[deltaQ_pred, new_state_h, new_state_c],
    #         name="decoder_model"
    #     )
    #
    #     return decoder_model

    def build_decoder(self, max_decoder_timesteps: int, max_transitions: int, bins: int, embedding_layer):
        """
        Build the decoder part of the Seq2Seq model using a shared embedding layer.
        """
        embedding_dim = self.config['model_config']['embedding_size']
        latent_state_neuron = self.config['model_config']['latent_state_neuron']

        # Decoder Inputs (Per-Timestep)
        decoder_bin_indices = keras.layers.Input(
            shape=(max_decoder_timesteps, max_transitions), name="decoder_bin_indices", dtype='int32'
        )

        decoder_time_spent = keras.layers.Input(
            shape=(max_decoder_timesteps, max_transitions), name="decoder_time_spent", dtype="float32"
        )

        decoder_previous_Q = keras.layers.Input(
            shape=(max_decoder_timesteps), name="decoder_previous_Q", dtype="float32"
        )

        # Decoder Initial States
        decoder_state_h = keras.layers.Input(shape=(latent_state_neuron,), name="decoder_state_h", dtype="float32")
        decoder_state_c = keras.layers.Input(shape=(latent_state_neuron,), name="decoder_state_c", dtype="float32")

        # Use the shared embedding layer
        decoder_embedding = embedding_layer(
            decoder_bin_indices)  # Shape: (batch_size, max_decoder_timesteps, max_transitions, embedding_dim)

        # Add an extra dimension to decoder_time_spent
        decoder_time_spent_expanded = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1)
        )(decoder_time_spent)  # Shape: (batch_size, max_decoder_timesteps, max_transitions, 1)

        # Concatenate decoder inputs (along the last axis)
        decoder_interaction = keras.layers.Concatenate(axis=-1)(
            [decoder_embedding, decoder_time_spent_expanded]
        )  # Shape: (batch_size, max_decoder_timesteps, max_transitions, embedding_dim + 1)

        # Adjusted Convolutional Layers for Decoder
        decoder_features = keras.Sequential([
            keras.layers.Conv1D(filters=2, kernel_size=15, strides=4, activation='tanh', name='Decoder_Convo1'),
            keras.layers.Conv1D(filters=4, kernel_size=25, strides=4, activation='tanh', name='Decoder_Convo2'),
            keras.layers.Conv1D(filters=6, kernel_size=35, strides=4, activation='tanh', name='Decoder_Convo3'),
            keras.layers.Conv1D(filters=8, kernel_size=45, strides=5, activation='tanh', name='Decoder_Convo4'),
            keras.layers.Flatten(),
        ], name="decoder_convolution")(decoder_interaction)  # Shape: (batch_size, features)

        # Concatenate decoder_features with decoder_previous_Q
        decoder_features_and_previous_cap = keras.layers.Concatenate()(
            [decoder_features, decoder_previous_Q]
        )  # Shape: (batch_size, features + 1)

        # Reshape for LSTM
        decoder_features_and_previous_cap_expanded = keras.layers.Reshape((1, -1))(
            decoder_features_and_previous_cap
        )  # Shape: (batch_size, 1, features + 1)

        # Decoder LSTM Layer with initial states
        decoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron, return_sequences=True, return_state=True, name="decoder_lstm"
        )
        decoder_outputs, new_state_h, new_state_c = decoder_lstm(
            decoder_features_and_previous_cap_expanded, initial_state=[decoder_state_h, decoder_state_c]
        )  # Shape: (batch_size, 1, latent_state_neuron)

        # Output Layer
        deltaQ_pred = keras.layers.Dense(1, activation="linear", name="deltaQ_output")(
            decoder_outputs)  # Shape: (batch_size, 1, 1)

        deltaQ_pred = keras.layers.Flatten()(deltaQ_pred)  # Shape: (batch_size, 1)

        # Define the decoder model with outputs including new states
        decoder_model = keras.models.Model(
            inputs=[
                decoder_bin_indices,
                decoder_time_spent,
                decoder_previous_Q,
                decoder_state_h,
                decoder_state_c
            ],
            outputs=[deltaQ_pred, new_state_h, new_state_c],
            name="decoder_model"
        )

        return decoder_model

    def build_full_sequence_to_sequence_with_previous_capacity(self, max_encoder_timesteps: int,
                                                                 max_decoder_timesteps: int,
                                                                 max_transitions: int,
                                                                 bins: int):
        """
        Build the full Seq2Seq model with shared embedding and step-by-step decoding.
        """
        # Create shared embedding layer
        self.embedding_layer = self.create_shared_embedding(bins)
        self.encoder_model = self.build_encoder(max_encoder_timesteps, max_transitions, bins, self.embedding_layer)
        self.decoder_model = self.build_decoder(1, max_transitions, bins, self.embedding_layer)


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

    def pad_and_mask_data(self, data_split, max_encoder_timesteps, max_decoder_timesteps, max_transitions=1,
                          pre_pad_encoder=False):
        """
        Pad transitions to max_transitions, then pad encoder and decoder sequences to max_encoder_timesteps and max_decoder_timesteps respectively.

        data_split: dictionary with keys 'encoder_inputs', 'decoder_inputs', 'decoder_outputs'.
        Returns: (inputs, targets) tuple of padded arrays ready for model input.
        """

        # If scalar_vector_exp_flag is 'table' or 'cube'
        if self.config['scalar_vector_exp_flag'] == 'table' or self.config['scalar_vector_exp_flag'] == 'cube':

            enc_time_spent = data_split['encoder_inputs']['timebins']
            dec_time_spent = data_split['decoder_inputs']['timebins']
            enc_cumulative_time_spent = data_split['encoder_inputs']['cumulative_time']
            dec_cumulative_time_spent = data_split['decoder_inputs']['cumulative_time']
            enc_Ah_throughput = data_split['encoder_inputs']['cumulative_Ah']
            dec_Ah_throughput = data_split['decoder_inputs']['cumulative_Ah']

            # converting to list of lists
            enc_time_spent = [[list(each_intermittent_cycle.values())
                               for each_intermittent_cycle in each_sequence]
                              for each_sequence in enc_time_spent]
            dec_time_spent = [[list(each_intermittent_cycle.values())
                               for each_intermittent_cycle in each_sequence]
                              for each_sequence in dec_time_spent]


            # Now pad along the timesteps dimension
            enc_padding_type = 'pre'

            # enc_bin_indices_padded = pad_sequences(enc_bin_indices, maxlen=max_encoder_timesteps,
            #                                        padding=enc_padding_type,
            #                                        value=0, dtype='int32')
            enc_time_spent_padded = pad_sequences(enc_time_spent, maxlen=max_encoder_timesteps,
                                                  padding=enc_padding_type,
                                                  value=0.0, dtype='float32')
            enc_capacity_diff_padded = pad_sequences(data_split['encoder_inputs']['capacity_difference'],
                                                     maxlen=max_encoder_timesteps, padding=enc_padding_type,
                                                     value=0.0, dtype='float32')
            enc_cumulative_time_spent_padded = pad_sequences(enc_cumulative_time_spent, maxlen=max_encoder_timesteps,
                                                             padding=enc_padding_type,
                                                             value=0.0, dtype='float32')
            enc_cumulative_Ah_throughput_padded = pad_sequences(enc_Ah_throughput, maxlen=max_encoder_timesteps,
                                                                padding=enc_padding_type,
                                                                value=0.0, dtype='float32')

            # reshape encoder_capacity_diff_padded to (batch_size, max_encoder_timesteps, 1)
            enc_capacity_diff_padded = np.expand_dims(enc_capacity_diff_padded, axis=-1)
            enc_capacity_diff_padded = enc_capacity_diff_padded.astype('float32')

            enc_cumulative_time_spent_padded = np.expand_dims(enc_cumulative_time_spent_padded, axis=-1)
            enc_cumulative_time_spent_padded = enc_cumulative_time_spent_padded.astype('float32')

            enc_cumulative_Ah_throughput_padded = np.expand_dims(enc_cumulative_Ah_throughput_padded, axis=-1)
            enc_cumulative_Ah_throughput_padded = enc_cumulative_Ah_throughput_padded.astype('float32')

            dec_padding_type = 'post'

            # dec_bin_indices_padded = pad_sequences(dec_bin_indices, maxlen=max_decoder_timesteps,
            #                                        padding=dec_padding_type, value=0, dtype='int32')
            dec_time_spent_padded = pad_sequences(dec_time_spent, maxlen=max_decoder_timesteps,
                                                  padding=dec_padding_type, value=0.0, dtype='float32')
            dec_targets_padded = pad_sequences(data_split['decoder_outputs']['capacity_difference'],
                                               maxlen=max_decoder_timesteps, padding=dec_padding_type,
                                               value=0.0, dtype='float32')
            dec_cumulative_time_spent_padded = pad_sequences(dec_cumulative_time_spent, maxlen=max_decoder_timesteps,
                                                                padding=dec_padding_type,
                                                                value=0.0, dtype='float32')
            dec_cumulative_Ah_throughput_padded = pad_sequences(dec_Ah_throughput, maxlen=max_decoder_timesteps,
                                                                padding=dec_padding_type,
                                                                value=0.0, dtype='float32')


            # Reshape targets to (batch_size, max_decoder_timesteps, 1)
            dec_targets_padded = np.expand_dims(dec_targets_padded, axis=-1)
            dec_targets_padded = dec_targets_padded.astype('float32')

            # Reshape cumulative_time_spent_padded to (batch_size, max_decoder_timesteps, 1)
            dec_cumulative_time_spent_padded = np.expand_dims(dec_cumulative_time_spent_padded, axis=-1)
            dec_cumulative_time_spent_padded = dec_cumulative_time_spent_padded.astype('float32')

            dec_cumulative_Ah_throughput_padded = np.expand_dims(dec_cumulative_Ah_throughput_padded, axis=-1)
            dec_cumulative_Ah_throughput_padded = dec_cumulative_Ah_throughput_padded.astype('float32')

            inputs = {
                'encoder_time_spent': tf.convert_to_tensor(enc_time_spent_padded),
                'encoder_capacity_delta': tf.convert_to_tensor(enc_capacity_diff_padded),
                'encoder_cumulative_time_spent': tf.convert_to_tensor(enc_cumulative_time_spent_padded),
                'encoder_cumulative_Ah_throughput': tf.convert_to_tensor(enc_cumulative_Ah_throughput_padded),
                'decoder_time_spent': tf.convert_to_tensor(dec_time_spent_padded),
                'decoder_cumulative_time_spent': tf.convert_to_tensor(dec_cumulative_time_spent_padded),
                'decoder_cumulative_Ah_throughput': tf.convert_to_tensor(dec_cumulative_Ah_throughput_padded)
            }

            targets = tf.convert_to_tensor(dec_targets_padded, dtype='float32')
            q_initial = tf.convert_to_tensor(
                [each_sequence[0] for each_sequence in data_split['decoder_inputs']['previous_capacity']],
                dtype='float32')
            q_initial_expanded = tf.expand_dims(q_initial, axis=1)  # Shape: (batch_size, 1)
            # q_initial_expanded = tf.expand_dims(q_initial_expanded, axis=-1)  # Shape: (batch_size,1,1)

            return inputs, targets, q_initial_expanded

        else: # scalar_vector_exp_flag == 'matrix'

            # Encoder
            enc_bin_indices = self.pad_transitions(data_split['encoder_inputs']['bin_indices'], max_transitions)
            enc_time_spent = self.pad_transitions(data_split['encoder_inputs']['timebins'], max_transitions)
            enc_cumulative_time_spent = data_split['encoder_inputs']['cumulative_time']
            enc_cumulative_Ah_throughput = data_split['encoder_inputs']['cumulative_Ah']
            # enc_capacity_diff = self.pad_transitions(data_split['encoder_inputs']['capacity_difference'], max_transitions)

            # Decoder
            dec_bin_indices = self.pad_transitions(data_split['decoder_inputs']['bin_indices'], max_transitions)
            dec_time_spent = self.pad_transitions(data_split['decoder_inputs']['timebins'], max_transitions)
            dec_cumulative_time_spent = data_split['decoder_inputs']['cumulative_time']
            dec_cumulative_Ah_throughput = data_split['decoder_inputs']['cumulative_Ah']
            # dec_targets = self.pad_transitions(data_split['decoder_outputs']['capacity_difference'], max_transitions)

            # Now pad along the timesteps dimension
            enc_padding_type = 'pre'

            enc_bin_indices_padded = pad_sequences(enc_bin_indices, maxlen=max_encoder_timesteps,
                                                   padding=enc_padding_type,
                                                   value=0, dtype='int32')
            enc_time_spent_padded = pad_sequences(enc_time_spent, maxlen=max_encoder_timesteps,
                                                  padding=enc_padding_type,
                                                  value=0.0, dtype='float32')
            enc_capacity_diff_padded = pad_sequences(data_split['encoder_inputs']['capacity_difference'],
                                                     maxlen=max_encoder_timesteps, padding=enc_padding_type,
                                                     value=0.0, dtype='float32')
            enc_cumulative_time_spent_padded = pad_sequences(enc_cumulative_time_spent, maxlen=max_encoder_timesteps,
                                                                padding=enc_padding_type,
                                                                value=0.0, dtype='float32')
            enc_cumulative_Ah_throughput_padded = pad_sequences(enc_cumulative_Ah_throughput, maxlen=max_encoder_timesteps,
                                                                padding=enc_padding_type,
                                                                value=0.0, dtype='float32')


            # reshape encoder_capacity_diff_padded to (batch_size, max_encoder_timesteps, 1)
            enc_capacity_diff_padded = np.expand_dims(enc_capacity_diff_padded, axis=-1)
            enc_capacity_diff_padded = enc_capacity_diff_padded.astype('float32')

            enc_cumulative_time_spent_padded = np.expand_dims(enc_cumulative_time_spent_padded, axis=-1)
            enc_cumulative_time_spent_padded = enc_cumulative_time_spent_padded.astype('float32')

            enc_cumulative_Ah_throughput_padded = np.expand_dims(enc_cumulative_Ah_throughput_padded, axis=-1)
            enc_cumulative_Ah_throughput_padded = enc_cumulative_Ah_throughput_padded.astype('float32')

            dec_padding_type = 'post'

            dec_bin_indices_padded = pad_sequences(dec_bin_indices, maxlen=max_decoder_timesteps,
                                                   padding=dec_padding_type, value=0, dtype='int32')
            dec_time_spent_padded = pad_sequences(dec_time_spent, maxlen=max_decoder_timesteps,
                                                  padding=dec_padding_type, value=0.0, dtype='float32')
            dec_targets_padded = pad_sequences(data_split['decoder_outputs']['capacity_difference'],
                                               maxlen=max_decoder_timesteps, padding=dec_padding_type,
                                               value=0.0, dtype='float32')
            dec_cumulative_time_spent_padded = pad_sequences(dec_cumulative_time_spent, maxlen=max_decoder_timesteps,
                                                                padding=dec_padding_type,
                                                                value=0.0, dtype='float32')
            dec_cumulative_Ah_throughput_padded = pad_sequences(dec_cumulative_Ah_throughput, maxlen=max_decoder_timesteps,
                                                                padding=dec_padding_type,
                                                                value=0.0, dtype='float32')

            # Reshape targets to (batch_size, max_decoder_timesteps, 1)
            dec_targets_padded = np.expand_dims(dec_targets_padded, axis=-1)
            dec_targets_padded = dec_targets_padded.astype('float32')

            # Reshape cumulative_time_spent_padded to (batch_size, max_decoder_timesteps, 1)
            dec_cumulative_time_spent_padded = np.expand_dims(dec_cumulative_time_spent_padded, axis=-1)
            dec_cumulative_time_spent_padded = dec_cumulative_time_spent_padded.astype('float32')

            # Reshape cumulative_Ah_throughput_padded to (batch_size, max_decoder_timesteps, 1)
            dec_cumulative_Ah_throughput_padded = np.expand_dims(dec_cumulative_Ah_throughput_padded, axis=-1)
            dec_cumulative_Ah_throughput_padded = dec_cumulative_Ah_throughput_padded.astype('float32')

            inputs = {
                'encoder_bin_indices': tf.convert_to_tensor(enc_bin_indices_padded),
                'encoder_time_spent': tf.convert_to_tensor(enc_time_spent_padded),
                'encoder_capacity_delta': tf.convert_to_tensor(enc_capacity_diff_padded),
                'encoder_cumulative_time_spent': tf.convert_to_tensor(enc_cumulative_time_spent_padded),
                'encoder_cumulative_Ah_throughput': tf.convert_to_tensor(enc_cumulative_Ah_throughput_padded),
                'decoder_bin_indices': tf.convert_to_tensor(dec_bin_indices_padded),
                'decoder_time_spent': tf.convert_to_tensor(dec_time_spent_padded),
                'decoder_cumulative_time_spent': tf.convert_to_tensor(dec_cumulative_time_spent_padded),
                'decoder_cumulative_Ah_throughput': tf.convert_to_tensor(dec_cumulative_Ah_throughput_padded)
            }

            targets = tf.convert_to_tensor(dec_targets_padded, dtype='float32')
            q_initial = tf.convert_to_tensor(
                [each_sequence[0] for each_sequence in data_split['decoder_inputs']['previous_capacity']],
                dtype='float32')
            q_initial_expanded = tf.expand_dims(q_initial, axis=1)  # Shape: (batch_size, 1)
            # q_initial_expanded = tf.expand_dims(q_initial_expanded, axis=-1)  # Shape: (batch_size,1,1)

            return inputs, targets, q_initial_expanded


    def train_sequence_to_sequence_model_with_table_or_cube_data(self, data: Dict[str, Any],max_encoder_timesteps: int, max_decoder_timesteps: int):
        """
        Train the sequence-to-sequence model using provided training and validation data with two-level padding and masking.
        """

        # Preprocess and pad training and validation data
        train_inputs, train_targets, _ = self.pad_and_mask_data(data['train'], max_encoder_timesteps, max_decoder_timesteps, pre_pad_encoder=False)
        val_inputs, val_targets, _ = self.pad_and_mask_data(data['validation'], max_encoder_timesteps, max_decoder_timesteps, pre_pad_encoder=False)

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

    def train_sequence_to_sequence_model_with_matrix_data(self, data: Dict[str, Any], max_encoder_timesteps: int, max_decoder_timesteps: int, max_transitions: int):
        """
        Train the sequence-to-sequence model using provided training and validation data with two-level padding and masking.
        """

        # Preprocess and pad training and validation data
        train_inputs, train_targets, _ = self.pad_and_mask_data(data['train'], max_encoder_timesteps, max_decoder_timesteps, max_transitions, pre_pad_encoder=False)
        val_inputs, val_targets, _ = self.pad_and_mask_data(data['validation'], max_encoder_timesteps, max_decoder_timesteps, max_transitions, pre_pad_encoder=False)

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

    def setup_optimizer_and_loss(self):
        """Set up optimizer with mixed precision and define the loss function."""
        base_optimizer = keras.optimizers.Adam(
            learning_rate=self.config['model_specifications']['learning_rate']
        )
        self.optimizer = mixed_precision.LossScaleOptimizer(base_optimizer, dynamic=True)
        self.loss_fn = keras.losses.get(self.config['model_config']['type_of_loss'])
        print("Optimizer and loss function set up.")

    def setup_metrics(self):
        """Initialize training and validation metrics."""
        self.train_mae = keras.metrics.MeanAbsoluteError(name='train_mae')
        self.train_mse = keras.metrics.MeanSquaredError(name='train_mse')
        self.val_mae = keras.metrics.MeanAbsoluteError(name='val_mae')
        self.val_mse = keras.metrics.MeanSquaredError(name='val_mse')
        print("Metrics initialized.")

    def setup_mixed_precision(self):
        """Enable mixed precision for faster training."""
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"Mixed precision policy set to: {policy.name}")

    @tf.function
    def train_step(self, encoder_bin, encoder_time, encoder_delta,
                   decoder_bin, decoder_time_spent, decoder_target,
                   Q_initial, max_decoder_timesteps):
        """
        Perform a single training step.

        :param encoder_bin: Tensor of encoder bin indices. Shape: (batch_size, max_encoder_timesteps)
        :param encoder_time: Tensor of encoder time spent. Shape: (batch_size, max_encoder_timesteps)
        :param encoder_delta: Tensor of encoder capacity delta. Shape: (batch_size, max_encoder_timesteps, 1)
        :param decoder_bin: Tensor of decoder bin indices. Shape: (batch_size, max_decoder_timesteps, max_transitions)
        :param decoder_time_spent: Tensor of decoder time spent. Shape: (batch_size, max_decoder_timesteps, max_transitions)
        :param decoder_target: Tensor of decoder targets. Shape: (batch_size, max_decoder_timesteps, 1)
        :param Q_initial: Tensor of initial Q values. Shape: (batch_size, 1, 1)
        :param max_decoder_timesteps: Maximum number of timesteps for the decoder.
        :return: Computed loss for the batch.
        """
        with tf.GradientTape() as tape:
            # Encoder Forward Pass
            encoder_states = self.encoder_model([encoder_bin, encoder_time, encoder_delta], training=True)
            decoder_state_h, decoder_state_c = encoder_states

            # Initialize previous_Q with Q_initial
            previous_Q = Q_initial  # Shape: (batch_size, 1)

            # Initialize TensorArray to collect predictions
            predictions = tf.TensorArray(dtype=tf.float32, size=max_decoder_timesteps)

            # Vectorized decoder loop using tf.range
            for t in tf.range(max_decoder_timesteps):
                # Current timestep inputs
                current_decoder_bin = decoder_bin[:, t:t + 1, :]  # Shape: (batch_size, 1, max_transitions)
                current_decoder_time_spent = decoder_time_spent[:, t:t + 1,
                                             :]  # Shape: (batch_size, 1, max_transitions)

                # Pass to decoder
                deltaQ_pred, new_state_h, new_state_c = self.decoder_model([
                    current_decoder_bin,
                    current_decoder_time_spent,
                    previous_Q,
                    decoder_state_h,
                    decoder_state_c
                ], training=True)  # deltaQ_pred: (batch_size, 1, 1)

                # Verify deltaQ_pred shape
                tf.debugging.assert_rank(deltaQ_pred, 2, message="deltaQ_pred must be 3D [batch_size, 1, 1]")
                tf.debugging.assert_equal(tf.shape(deltaQ_pred)[1], 1, message="deltaQ_pred must have time_steps=1")


                # Write prediction to TensorArray
                predictions = predictions.write(t, deltaQ_pred)

                # Update Q for the next timestep
                previous_Q = previous_Q + deltaQ_pred  # Q = Q_initial + ΔQ_pred

                # Update decoder states
                decoder_state_h, decoder_state_c = new_state_h, new_state_c

            # Stack all predictions
            predictions = predictions.stack()  # Shape: (max_decoder_timesteps, batch_size, 1, 1)
            predictions = tf.transpose(predictions,
                                       perm=[1, 0, 2])  # Shape: (batch_size, max_decoder_timesteps, 1, 1)
            # predictions = tf.squeeze(predictions, axis=-1)  # Shape: (batch_size, max_decoder_timesteps, 1)

            # Compute loss
            loss = self.loss_fn(decoder_target, predictions)  # Shapes: [batch_size, max_decoder_timesteps, 1]

        # Compute gradients
        gradients = tape.gradient(loss,
                                  self.encoder_model.trainable_variables + self.decoder_model.trainable_variables)

        # Apply gradients
        self.optimizer.apply_gradients(
            zip(gradients, self.encoder_model.trainable_variables + self.decoder_model.trainable_variables)
        )

        # Update metrics
        self.train_mae.update_state(decoder_target, predictions)
        self.train_mse.update_state(decoder_target, predictions)

        return loss

    @tf.function
    def val_step(self, encoder_bin, encoder_time, encoder_delta,
                 decoder_bin, decoder_time_spent, decoder_target,
                 Q_initial, max_decoder_timesteps):
        """
        Perform a single validation step.

        :param encoder_bin: Tensor of encoder bin indices.
        :param encoder_time: Tensor of encoder time spent.
        :param encoder_delta: Tensor of encoder capacity delta.
        :param decoder_bin: Tensor of decoder bin indices.
        :param decoder_time_spent: Tensor of decoder time spent.
        :param decoder_target: Tensor of decoder targets.
        :param Q_initial: Tensor of initial Q values.
        :param max_decoder_timesteps: Maximum number of timesteps for the decoder.
        :return: Computed validation loss for the batch.
        """
        # Encoder Forward Pass
        encoder_states = self.encoder_model([encoder_bin, encoder_time, encoder_delta],
                                            training=False)  # [state_h, state_c]
        decoder_state_h, decoder_state_c = encoder_states

        # Initialize previous_Q with Q_initial
        previous_Q = Q_initial  # Shape: (batch_size, 1, 1)

        # Initialize TensorArray to collect predictions
        predictions = tf.TensorArray(dtype=tf.float32, size=max_decoder_timesteps)

        # Vectorized decoder loop using tf.range
        for t in tf.range(max_decoder_timesteps):
            # Current timestep inputs
            current_decoder_bin = decoder_bin[:, t:t + 1, :]  # Shape: (batch_size, 1, max_transitions)
            current_decoder_time_spent = decoder_time_spent[:, t:t + 1, :]  # Shape: (batch_size, 1, max_transitions)

            # Pass to decoder
            deltaQ_pred, new_state_h, new_state_c = self.decoder_model([
                current_decoder_bin,
                current_decoder_time_spent,
                previous_Q,
                decoder_state_h,
                decoder_state_c
            ], training=False)  # Shapes as specified

            # Write prediction to TensorArray
            predictions = predictions.write(t, deltaQ_pred)

            # Update Q for the next timestep
            previous_Q = previous_Q + deltaQ_pred  # Q = Q_initial + ΔQ_pred

            # Update decoder states
            decoder_state_h, decoder_state_c = new_state_h, new_state_c

        # Stack all predictions
        predictions = predictions.stack()  # Shape: (max_decoder_timesteps, batch_size, 1)
        predictions = tf.transpose(predictions,  perm=[1, 0, 2])  # Shape: (batch_size, max_decoder_timesteps, 1)

        # Compute loss
        val_loss = self.loss_fn(decoder_target, predictions)

        # Update metrics
        self.val_mae.update_state(decoder_target, predictions)
        self.val_mse.update_state(decoder_target, predictions)

        return val_loss

    def train_full_sequence_to_sequence_model_with_previous_capacity(self, data: Dict[str, Any],
                                                                     max_encoder_timesteps: int,
                                                                     max_decoder_timesteps: int,
                                                                     max_transitions: int):
        """
        Train the sequence-to-sequence model using provided training and validation data.
        Incorporates previous timestep's estimated capacity into the decoder without teacher forcing.

        :param data: Dictionary containing 'train' and 'validation' datasets.
        :param max_encoder_timesteps: Maximum number of timesteps for the encoder.
        :param max_decoder_timesteps: Maximum number of timesteps for the decoder.
        :param max_transitions: Maximum number of transitions in the bin matrix.
        """
        # Preprocess and pad training and validation data
        train_inputs, train_targets, train_decoder_first_timestep_prev_Q = self.pad_and_mask_data(
            data['train'], max_encoder_timesteps, max_decoder_timesteps, max_transitions
        )
        val_inputs, val_targets, val_decoder_first_timestep_prev_Q = self.pad_and_mask_data(
            data['validation'], max_encoder_timesteps, max_decoder_timesteps, max_transitions
        )

        # Convert numpy arrays to TensorFlow datasets
        batch_size = self.config['model_specifications']['batch_size']

        train_dataset = tf.data.Dataset.from_tensor_slices((
            train_inputs['encoder_bin_indices'],
            train_inputs['encoder_time_spent'],
            train_inputs['encoder_capacity_delta'],
            train_inputs['decoder_bin_indices'],
            train_inputs['decoder_time_spent'],
            train_targets,
            train_decoder_first_timestep_prev_Q
        )).shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            val_inputs['encoder_bin_indices'],
            val_inputs['encoder_time_spent'],
            val_inputs['encoder_capacity_delta'],
            val_inputs['decoder_bin_indices'],
            val_inputs['decoder_time_spent'],
            val_targets,
            val_decoder_first_timestep_prev_Q
        )).batch(batch_size, drop_remainder=True).prefetch(buffer_size=tf.data.AUTOTUNE)

        # Early Stopping Parameters
        patience = self.config['model_specifications']['early_stopping_patience']
        best_val_loss = np.inf
        wait = 0

        # Model Checkpoint Path
        checkpoint_path = os.path.join(self.model_dir, 'best_model')
        os.makedirs(checkpoint_path, exist_ok=True)

        # Training Loop
        epochs = self.config['model_specifications']['epochs']
        for epoch in range(epochs):

            time_begin = time.time()
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Reset metrics
            self.train_mae.reset_states()
            self.train_mse.reset_states()
            self.val_mae.reset_states()
            self.val_mse.reset_states()

            # ============================
            # Training Phase
            # ============================
            for step, (encoder_bin, encoder_time, encoder_delta,
                       decoder_bin, decoder_time_spent, decoder_target,
                       Q_initial) in enumerate(train_dataset):
                loss = self.train_step(encoder_bin, encoder_time, encoder_delta,
                                       decoder_bin, decoder_time_spent, decoder_target,
                                       Q_initial, max_decoder_timesteps)

                # if step % 100 == 0:
                #     print(f"Step {step}, Loss: {loss.numpy():.4f}")

            # ============================
            # Validation Phase
            # ============================
            for step, (encoder_bin, encoder_time, encoder_delta,
                       decoder_bin, decoder_time_spent, decoder_target,
                       Q_initial) in enumerate(val_dataset):
                self.val_step(encoder_bin, encoder_time, encoder_delta,
                              decoder_bin, decoder_time_spent, decoder_target,
                              Q_initial, max_decoder_timesteps)

            # ============================
            # Logging and Callbacks
            # ============================
            train_mae_result = self.train_mae.result().numpy()
            train_mse_result = self.train_mse.result().numpy()
            val_mae_result = self.val_mae.result().numpy()
            val_mse_result = self.val_mse.result().numpy()

            print(f"Train MAE: {train_mae_result:.6f}, Train MSE: {train_mse_result:.6f}")
            print(f"Val MAE: {val_mae_result:.6f}, Val MSE: {val_mse_result:.6f}")

            # Check for improvement
            if val_mse_result < best_val_loss:
                best_val_loss = val_mse_result
                wait = 0
                # Save the best model
                self.encoder_model.save(os.path.join(checkpoint_path, 'encoder_model'))
                self.decoder_model.save(os.path.join(checkpoint_path, 'decoder_model'))
                print("Validation loss improved. Saving models.")
            else:
                wait += 1
                print(f"No improvement in validation loss for {wait} epoch(s).")
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

            time_end = time.time()
            print(f"Epoch took {time_end - time_begin:.2f} seconds.")

        # Optionally, load the best models after training
        self.encoder_model = keras.models.load_model(os.path.join(checkpoint_path, 'encoder_model'), compile=False)
        self.decoder_model = keras.models.load_model(os.path.join(checkpoint_path, 'decoder_model'), compile=False)
        print("Best models loaded.")



    # def train_full_sequence_to_sequence_model_with_previous_capacity(self, data: Dict[str, Any],
    #                                                       max_encoder_timesteps: int,
    #                                                       max_decoder_timesteps: int,
    #                                                       max_transitions: int):
    #     """
    #     Train the sequence-to-sequence model using provided training and validation data.
    #     Incorporates previous timestep's estimated capacity into the decoder without teacher forcing.
    #
    #     :param data: Dictionary containing 'train' and 'validation' datasets.
    #     :param max_encoder_timesteps: Maximum number of timesteps for the encoder.
    #     :param max_decoder_timesteps: Maximum number of timesteps for the decoder.
    #     :param max_transitions: Maximum number of transitions in the bin matrix.
    #     """
    #     # Preprocess and pad training and validation data
    #     train_inputs, train_targets, train_decoder_first_timestep_prev_Q = self.pad_and_mask_data(
    #         data['train'], max_encoder_timesteps, max_decoder_timesteps, max_transitions
    #     )
    #     val_inputs, val_targets, val_decoder_first_timestep_prev_Q = self.pad_and_mask_data(
    #         data['validation'], max_encoder_timesteps, max_decoder_timesteps, max_transitions
    #     )
    #
    #     # Convert numpy arrays to TensorFlow datasets
    #     batch_size = self.config['model_specifications']['batch_size']
    #
    #     train_dataset = tf.data.Dataset.from_tensor_slices((
    #         train_inputs['encoder_bin_indices'],
    #         train_inputs['encoder_time_spent'],
    #         train_inputs['encoder_capacity_delta'],
    #         train_inputs['decoder_bin_indices'],
    #         train_inputs['decoder_time_spent'],
    #         train_targets,
    #         train_decoder_first_timestep_prev_Q
    #     )).shuffle(buffer_size=1024).batch(batch_size)
    #
    #     val_dataset = tf.data.Dataset.from_tensor_slices((
    #         val_inputs['encoder_bin_indices'],
    #         val_inputs['encoder_time_spent'],
    #         val_inputs['encoder_capacity_delta'],
    #         val_inputs['decoder_bin_indices'],
    #         val_inputs['decoder_time_spent'],
    #         val_targets,
    #         val_decoder_first_timestep_prev_Q
    #     )).batch(batch_size)
    #
    #     # Define optimizer and loss function
    #     optimizer = keras.optimizers.Adam(learning_rate=self.config['model_specifications']['learning_rate'])
    #     loss_fn = tf.keras.losses.get(self.config['model_config']['type_of_loss'])
    #
    #     # Define metrics
    #     train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
    #     train_mse = tf.keras.metrics.MeanSquaredError(name='train_mse')
    #
    #     val_mae = tf.keras.metrics.MeanAbsoluteError(name='val_mae')
    #     val_mse = tf.keras.metrics.MeanSquaredError(name='val_mse')
    #
    #     # Early Stopping Parameters
    #     patience = self.config['model_specifications']['early_stopping_patience']
    #     best_val_loss = np.inf
    #     wait = 0
    #
    #     # Model Checkpoint Path
    #     checkpoint_path = os.path.join(self.model_dir, 'best_model')
    #
    #     # Training Loop
    #     epochs = self.config['model_specifications']['epochs']
    #     for epoch in range(epochs):
    #
    #         time_begin = time.time()
    #         print(f"\nEpoch {epoch + 1}/{epochs}")
    #
    #         # Reset metrics
    #         train_mae.reset_states()
    #         train_mse.reset_states()
    #         val_mae.reset_states()
    #         val_mse.reset_states()
    #
    #         # ============================
    #         # Training Phase
    #         # ============================
    #         for step, (encoder_bin, encoder_time, encoder_delta,
    #                    decoder_bin, decoder_time_spent, decoder_target,
    #                    Q_initial) in enumerate(train_dataset):
    #             with tf.GradientTape() as tape:
    #                 # Encoder Forward Pass
    #                 encoder_states = self.encoder_model([encoder_bin, encoder_time, encoder_delta],
    #                                                     training=True)  # [state_h, state_c]
    #
    #                 # Initialize decoder states with encoder states
    #                 decoder_state_h, decoder_state_c = encoder_states
    #
    #                 # Initialize previous_Q with Q_initial
    #                 # Shape: (batch_size, 1, 1)
    #                 previous_Q = Q_initial  # [batch_size, 1, 1]
    #
    #                 # Collect predictions
    #                 predictions = []
    #
    #                 for t in range(max_decoder_timesteps):
    #                     # Current timestep inputs
    #                     current_decoder_bin = decoder_bin[:, t:t + 1, :]  # Shape: (batch_size, 1, max_transitions)
    #                     current_decoder_time_spent = decoder_time_spent[:, t:t + 1,
    #                                                  :]  # Shape: (batch_size, 1, max_transitions)
    #
    #                     # Pass to decoder
    #                     deltaQ_pred, new_state_h, new_state_c = self.decoder_model([
    #                         current_decoder_bin,
    #                         current_decoder_time_spent,
    #                         previous_Q,
    #                         decoder_state_h,
    #                         decoder_state_c
    #                     ],
    #                         training=True)  # Shape: (batch_size, 1, 1), (batch_size, latent_state_neuron), (batch_size, latent_state_neuron)
    #
    #                     # Append prediction
    #                     predictions.append(deltaQ_pred)
    #
    #                     # Update Q for the next timestep
    #                     previous_Q = previous_Q + deltaQ_pred  # Q = Q_initial + ΔQ_pred
    #
    #                     # Update decoder states
    #                     decoder_state_h, decoder_state_c = new_state_h, new_state_c
    #
    #                 # Concatenate all predictions
    #                 predictions = tf.concat(predictions, axis=1)  # Shape: (batch_size, max_decoder_timesteps, 1)
    #
    #                 # Compute loss
    #                 loss = loss_fn(decoder_target, predictions)
    #
    #             # Compute gradients
    #             gradients = tape.gradient(loss,
    #                                       self.encoder_model.trainable_variables + self.decoder_model.trainable_variables)
    #
    #             # Apply gradients
    #             optimizer.apply_gradients(
    #                 zip(gradients, self.encoder_model.trainable_variables + self.decoder_model.trainable_variables))
    #
    #             # Update metrics
    #             train_mae.update_state(decoder_target, predictions)
    #             train_mse.update_state(decoder_target, predictions)
    #
    #         # ============================
    #         # Validation Phase
    #         # ============================
    #         for step, (encoder_bin, encoder_time, encoder_delta,
    #                    decoder_bin, decoder_time_spent, decoder_target,
    #                    Q_initial) in enumerate(val_dataset):
    #             # Encoder Forward Pass
    #             encoder_states = self.encoder_model([encoder_bin, encoder_time, encoder_delta],
    #                                                 training=False)  # [state_h, state_c]
    #
    #             # Initialize decoder states with encoder states
    #             decoder_state_h, decoder_state_c = encoder_states
    #
    #             # Initialize previous_Q with Q_initial
    #             previous_Q = Q_initial  # [batch_size, 1, 1]
    #
    #             # Collect predictions
    #             predictions = []
    #
    #             for t in range(max_decoder_timesteps):
    #                 # Current timestep inputs
    #                 current_decoder_bin = decoder_bin[:, t:t + 1, :]  # Shape: (batch_size, 1, max_transitions)
    #                 current_decoder_time_spent = decoder_time_spent[:, t:t + 1,
    #                                              :]  # Shape: (batch_size, 1, max_transitions)
    #
    #                 # Pass to decoder
    #                 deltaQ_pred, new_state_h, new_state_c = self.decoder_model([
    #                     current_decoder_bin,
    #                     decoder_time_spent[:, t:t + 1, :],
    #                     previous_Q,
    #                     decoder_state_h,
    #                     decoder_state_c
    #                 ],
    #                     training=False)  # Shape: (batch_size, 1, 1), (batch_size, latent_state_neuron), (batch_size, latent_state_neuron)
    #
    #                 # Append prediction
    #                 predictions.append(deltaQ_pred)
    #
    #                 # Update Q for the next timestep
    #                 previous_Q = previous_Q + deltaQ_pred  # Q = Q_initial + ΔQ_pred
    #
    #                 # Update decoder states
    #                 decoder_state_h, decoder_state_c = new_state_h, new_state_c
    #
    #             # Concatenate all predictions
    #             predictions = tf.concat(predictions, axis=1)  # Shape: (batch_size, max_decoder_timesteps, 1)
    #
    #             # Compute loss
    #             val_loss = loss_fn(decoder_target, predictions)
    #
    #             # Update metrics
    #             val_mae.update_state(decoder_target, predictions)
    #             val_mse.update_state(decoder_target, predictions)
    #
    #         # ============================
    #         # Logging and Callbacks
    #         # ============================
    #         train_mae_result = train_mae.result().numpy()
    #         train_mse_result = train_mse.result().numpy()
    #         val_mae_result = val_mae.result().numpy()
    #         val_mse_result = val_mse.result().numpy()
    #
    #
    #         print(f"Train MAE: {train_mae_result:.9f}, Train MSE: {train_mse_result:.9f}")
    #         print(f"Val MAE: {val_mae_result:.9f}, Val MSE: {val_mse_result:.9f}")
    #
    #         # Check for improvement
    #         if val_mse_result < best_val_loss:
    #             best_val_loss = val_mse_result
    #             wait = 0
    #             # Save the best model
    #             self.encoder_model.save(os.path.join(checkpoint_path, 'encoder_model'))
    #             self.decoder_model.save(os.path.join(checkpoint_path, 'decoder_model'))
    #             print("Validation loss improved. Saving models.")
    #         else:
    #             wait += 1
    #             print(f"No improvement in validation loss for {wait} epoch(s).")
    #             if wait >= patience:
    #                 print("Early stopping triggered.")
    #                 break
    #
    #         time_end = time.time()
    #         print(f"Epoch took {time_end - time_begin:.2f} seconds.")
    #
    #     # Optionally, load the best models after training
    #     self.encoder_model = keras.models.load_model(os.path.join(checkpoint_path, 'encoder_model'))
    #     self.decoder_model = keras.models.load_model(os.path.join(checkpoint_path, 'decoder_model'))
    #
    # def prepare_training_data(self, dataset, max_encoder_timesteps, max_decoder_timesteps, max_transitions):
    #     """
    #     Prepares the training or validation data by padding and masking.
    #
    #     :param dataset: Dataset dictionary containing 'encoder_bin_indices', 'encoder_time_spent',
    #                    'encoder_capacity_delta', 'decoder_bin_indices', 'decoder_time_spent', 'decoder_target'.
    #     :param max_encoder_timesteps: Maximum encoder timesteps.
    #     :param max_decoder_timesteps: Maximum decoder timesteps.
    #     :param max_transitions: Maximum number of transitions.
    #     :return: Tuple of (inputs, targets, Q_initial)
    #     """
    #     # Extract data
    #     encoder_bin = dataset['encoder_bin_indices']
    #     encoder_time = dataset['encoder_time_spent']
    #     encoder_delta = dataset['encoder_capacity_delta']
    #     decoder_bin = dataset['decoder_bin_indices']
    #     decoder_time_spent = dataset['decoder_time_spent']
    #     decoder_target = dataset['decoder_target']
    #
    #     # Initialize Q_initial (e.g., set to zero or use the first encoder_capacity_delta)
    #     # Here, we use the first timestep's capacity delta as Q_initial
    #     Q_initial = encoder_delta[:, 0, :]  # Shape: (num_samples, 1)
    #
    #     # Return as dictionary for easier handling
    #     inputs = {
    #         'encoder_bin_indices': encoder_bin,
    #         'encoder_time_spent': encoder_time,
    #         'encoder_capacity_delta': encoder_delta,
    #         'decoder_bin_indices': decoder_bin,
    #         'decoder_time_spent': decoder_time_spent
    #     }
    #     targets = decoder_target
    #
    #     return inputs, targets, Q_initial
    #
    # def load_model(self, exp_path: str):
    #     """
    #     Load a saved model from the configured directory.
    #     """
    #     trained_model_path = os.path.join(exp_path, 'checkpoints', 'model.keras')
    #     # self.model = keras.models.load_model(trained_model_path)
    #     self.model.load_weights(trained_model_path)

    def ConvToFixedP(self, filters, kernel_sizes, strides, p, name=None):
        """
        Sequential block that:
          1) applies Conv1D stack with 'same' padding,
          2) adaptive pools to length p,
          3) transposes to (batch, channels, p).
        """
        seq = keras.Sequential(name=name)
        # 1) convolutional feature extraction (same padding to avoid negative dims)
        for i, (f, k, s) in enumerate(zip(filters, kernel_sizes, strides), 1):
            seq.add(keras.layers.Conv1D(
                filters=f,
                kernel_size=k,
                strides=s,
                padding='same',
                activation="relu",
                name=f"{name}_conv{i}"
            ))

        # 2) adaptive pooling to p segments
        def adaptive_pool(x):
            length = tf.shape(x)[1]
            # ensure at least p frames
            frame_len = tf.maximum(length // p, 1)
            total = frame_len * p
            x_trunc = x[:, :total, :]  # truncate to divisible length
            x_resh = tf.reshape(x_trunc, [tf.shape(x)[0], p, frame_len, tf.shape(x)[2]])
            pooled = tf.reduce_mean(x_resh, axis=2)  # (batch, p, channels)
            return tf.transpose(pooled, perm=[0, 2, 1])  # (batch, channels, p)


        seq.add(keras.layers.Lambda(adaptive_pool, name=f"{name}_adaptive_pool"))

        # 3) flatten the last two dims into a single feature vector
        seq.add(keras.layers.Flatten(name=f"{name}_flatten"))

        return seq

    def build_sequence_to_sequence_model_with_matrix_data_auto(self,
                                                          max_encoder_timesteps: int,
                                                          max_decoder_timesteps: int,
                                                          max_transitions: int,
                                                          bins: int):
        """
        Build an Encoder-Decoder Seq2Seq model with variable Conv1D kernels/strides
        and guaranteed output shape (..., 8, p).
        """
        # Config
        embedding_dim = self.config['model_config']['embedding_size']
        type_of_loss = self.config['model_config']['type_of_loss']
        learning_rate = self.config['model_specifications']['learning_rate']
        latent_state_neuron = self.config['model_config']['latent_state_neuron']

        # Convolution parameters
        conv_filters = [2, 4, 6, 8]  # last must be 8
        conv_kernels = [15, 25, 35, 45]
        conv_strides = [4, 4, 4, 5]
        p = 3  # desired fixed temporal length

        # 1. Encoder Inputs
        encoder_bin_indices = keras.layers.Input(
            shape=(max_encoder_timesteps, max_transitions), name="encoder_bin_indices"
        )
        encoder_time_spent = keras.layers.Input(
            shape=(max_encoder_timesteps, max_transitions), name="encoder_time_spent"
        )
        encoder_capacity_delta = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_capacity_delta"
        )
        encoder_cumulative_time_spent = keras.layers.Input(
            shape=(max_encoder_timesteps, 1), name="encoder_cumulative_time_spent"
        )

        # Embedding + interaction
        timebin_encoding_layer = keras.layers.TimeDistributed(
            keras.layers.Embedding(input_dim=bins + 1, output_dim=embedding_dim, mask_zero=True),
            name="encoder_embedding"
        )
        enc_bin_emb = timebin_encoding_layer(encoder_bin_indices)
        enc_time_exp = keras.layers.Reshape((max_encoder_timesteps, max_transitions, 1))(encoder_time_spent)
        encoder_interaction = keras.layers.Concatenate()([enc_bin_emb, enc_time_exp])

        # Encoder convolution -> fixed (8, p)
        encoder_convolver = keras.layers.TimeDistributed(
            self.ConvToFixedP(conv_filters, conv_kernels, conv_strides, p, name="encoder_conv_block"),
            name="encoder_convolution"
        )

        encoder_features = encoder_convolver(encoder_interaction)
        # encoder_features.shape == (batch, max_encoder_timesteps, 8, p)

        # Combine and LSTM encoder
        encoder_combined = keras.layers.Concatenate(name="encoder_feature_concat")(
            [encoder_features, encoder_capacity_delta, encoder_cumulative_time_spent]
        )
        encoder_lstm = keras.layers.LSTM(units=latent_state_neuron, return_state=True, name="encoder_lstm")
        _, state_h, state_c = encoder_lstm(encoder_combined)

        # 2. Decoder Inputs
        decoder_bin_indices = keras.layers.Input(
            shape=(max_decoder_timesteps, max_transitions), name="decoder_bin_indices"
        )
        decoder_time_spent = keras.layers.Input(
            shape=(max_decoder_timesteps, max_transitions), name="decoder_time_spent"
        )
        decoder_cumulative_time_spent = keras.layers.Input(
            shape=(max_decoder_timesteps, 1), name="decoder_cumulative_time_spent"
        )

        dec_bin_emb = timebin_encoding_layer(decoder_bin_indices)
        dec_time_exp = keras.layers.Reshape((max_decoder_timesteps, max_transitions, 1))(decoder_time_spent)
        decoder_interaction = keras.layers.Concatenate()([dec_bin_emb, dec_time_exp])

        # Decoder convolution -> fixed (8, p)
        decoder_convolver = keras.layers.TimeDistributed(
            self.ConvToFixedP(conv_filters, conv_kernels, conv_strides, p, name="decoder_conv_block"),
            name="decoder_convolution"
        )
        decoder_features = decoder_convolver(decoder_interaction)
        # decoder_features.shape == (batch, max_decoder_timesteps, 8, p)


        # LSTM decoder
        decoder_combined = keras.layers.Concatenate(name="decoder_feature_concat")(
            [decoder_features, decoder_cumulative_time_spent]
        )
        decoder_lstm = keras.layers.LSTM(
            units=latent_state_neuron, return_sequences=True, name="decoder_lstm"
        )
        decoder_outputs = decoder_lstm(decoder_combined, initial_state=[state_h, state_c])

        # Output
        deltaQ_pred = keras.layers.TimeDistributed(
            keras.layers.Dense(1), name="deltaQ_output"
        )(decoder_outputs)

        model = keras.models.Model(
            inputs=[
                encoder_bin_indices, encoder_time_spent, encoder_capacity_delta, encoder_cumulative_time_spent,
                decoder_bin_indices, decoder_time_spent, decoder_cumulative_time_spent
            ],
            outputs=deltaQ_pred
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=type_of_loss,
            metrics=["mean_absolute_error", "mean_squared_error"]
        )

        print(model.summary(
            line_length=120,
            expand_nested=True,
            show_trainable=True
        ))

        self.model = model
        return model
