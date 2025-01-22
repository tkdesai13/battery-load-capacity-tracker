import os
import json
import random
import logging
import warnings

# Set TF logging level environment variable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

from data.data_processing import DataProcessor
from models.model import Model
from plot.model_evaluation import ModelEvaluator

# -----------------------------------------------------------------------------
# Global Configuration
# -----------------------------------------------------------------------------
# 1. Filter out python warnings
warnings.filterwarnings("ignore")

# 2. Set TensorFlow logger to only show errors
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def set_seeds(seed: int = 20) -> None:
    """
    Set random seeds for reproducibility across numpy, tensorflow, and random.

    Args:
        seed (int): Seed value. Default is 20.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def load_config(config_path: str = 'config.json') -> dict:
    """
    Load configuration data from a JSON file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)
    return config


def create_experiment_dir(config: dict) -> str:
    """
    Create a directory for experiment results if it doesn't already exist.

    Args:
        config (dict): Configuration dictionary (must contain 'paths' key with
                       'result' subkey, and 'experiment_name').

    Returns:
        str: Path to the experiment result directory.
    """
    experiment_result_dir = os.path.join(
        config['paths']['result'],
        config['experiment_name']
    )
    if not os.path.exists(experiment_result_dir):
        os.makedirs(experiment_result_dir)
    return experiment_result_dir


def make_serializable(data):
    """
    Recursively convert numpy arrays to lists and handle non-serializable keys.

    Args:
        data: Any data structure (dict, list, np.ndarray, etc.).

    Returns:
        Data converted into serializable form (lists, standard dictionaries, etc.).
    """
    if isinstance(data, dict):
        return {str(key): make_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [make_serializable(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main entry point for training, evaluating, and saving the experiment results
    based on the given configuration.
    """
    # Set seeds for reproducibility.
    set_seeds()

    # Load configuration and create result directory.
    config = load_config()
    experiment_result_dir = create_experiment_dir(config)

    # Initialize data processing.
    data_processing = DataProcessor(config=config)
    time_bins_and_transition_data = data_processing.load_and_binning_data()

    # Prepare features based on scalar/vector experiment flag.
    if config['scalar_vector_exp_flag'] in ('table', 'cube'):
        time_bins_and_transition_data, min_max_values = \
            data_processing.prepare_scalar_features(time_bins_and_transition_data)
    else:
        # Default to vector/embedded features if not 'table' or 'cube'.
        time_bins_and_transition_data, min_max_values = \
            data_processing.prepare_vector_features_embedded(time_bins_and_transition_data)

    # Scale data and split into training and test sets.
    scaled_data = data_processing.scale_data(time_bins_and_transition_data, min_max_values)

    # Sequence-to-sequence data preparation vs. standard.
    if config['sequence_to_sequence']:
        train_val_test_data, test_cells_id = \
            data_processing.prepare_data_sequence_to_sequence(scaled_data)
        shuffled_and_split_data, seq_lengths = \
            data_processing.shuffle_and_split_data_sequence_to_sequence(train_val_test_data)
    else:
        shuffled_and_split_data, test_cells_id = \
            data_processing.shuffle_and_split_data(scaled_data)

    # Initialize model and evaluator.
    model = Model(config=config)
    model_evaluator = ModelEvaluator(config=config, model_instance=model)

    # If a trained model path is provided and the trained_model flag is True, load it.
    if config['model_config']['trained_model'] and config['paths']['trained_model'] != '':
        max_sequence_length = int(min_max_values['max_sequence_length'])
        no_of_input_features = len(data_processing.bin_mapping)

        # Build the model architecture so the weights can be loaded.
        model.build_vector_model_TCN(
            max_sequence_length=max_sequence_length,
            no_of_input_features=no_of_input_features
        )
        # Load weights.
        model.load_model(exp_path=config['paths']['trained_model'])

        # Evaluate the loaded model.
        model_evaluator.evaluate_model(
            model=model.model,
            data=scaled_data,
            min_max_values=min_max_values,
            max_sequence_length=max_sequence_length,
            test_cells=test_cells_id
        )

    else:
        # Build, train, and evaluate a new model.
        if config['scalar_vector_exp_flag'] in ('table', 'cube'):
            no_of_input_features = len(data_processing.bin_mapping)

            if config['sequence_to_sequence']:
                # Sequence-to-sequence scenario with table or cube data.
                max_encoder_timesteps = int(
                    max([data["encoder_max"] for data_type, data in seq_lengths.items()])
                )
                max_decoder_timesteps = int(
                    max([data["decoder_max"] for data_type, data in seq_lengths.items()])
                )

                model.build_sequence_to_sequence_model_with_table_or_cube_data(
                    max_encoder_timesteps=max_encoder_timesteps,
                    max_decoder_timesteps=max_decoder_timesteps,
                    bins=no_of_input_features
                )

                model.train_sequence_to_sequence_model_with_table_or_cube_data(
                    data=shuffled_and_split_data,
                    max_encoder_timesteps=max_encoder_timesteps,
                    max_decoder_timesteps=max_decoder_timesteps
                )

                model_evaluator.evaluate_sequence_to_sequence_model(
                    model=model,
                    train_val_test_data=train_val_test_data,
                    max_encoder_timesteps=max_encoder_timesteps,
                    max_decoder_timesteps=max_decoder_timesteps
                )

            else:
                # Standard (non-sequence-to-sequence) scenario with table or cube data.
                no_of_input_features = len(shuffled_and_split_data['train']['timebins'][0])
                model.build_scalar_model(no_of_input_features=no_of_input_features)
                model.train_model(
                    train_data=shuffled_and_split_data['train'],
                    val_data=shuffled_and_split_data['validation']
                )
                model_evaluator.evaluate_model(
                    model=model.model,
                    data=scaled_data,
                    min_max_values=min_max_values,
                    test_cells=test_cells_id
                )

        elif config['scalar_vector_exp_flag'] == 'matrix':
            # Matrix-based data (vector + convolution).
            max_sequence_length = int(min_max_values['max_sequence_length'])
            no_of_input_features = len(data_processing.bin_mapping)

            if config['sequence_to_sequence']:
                max_encoder_timesteps = int(
                    max([data["encoder_max"] for data_type, data in seq_lengths.items()])
                )
                max_decoder_timesteps = int(
                    max([data["decoder_max"] for data_type, data in seq_lengths.items()])
                )

                # Build, train, and evaluate for sequence-to-sequence with matrix data.
                model.build_sequence_to_sequence_model_with_matrix_data(
                    max_encoder_timesteps=max_encoder_timesteps,
                    max_decoder_timesteps=max_decoder_timesteps,
                    max_transitions=max_sequence_length,
                    bins=no_of_input_features
                )
                model.train_sequence_to_sequence_model_with_matrix_data(
                    data=shuffled_and_split_data,
                    max_encoder_timesteps=max_encoder_timesteps,
                    max_decoder_timesteps=max_decoder_timesteps,
                    max_transitions=max_sequence_length
                )
                model_evaluator.evaluate_sequence_to_sequence_model(
                    model=model,
                    train_val_test_data=train_val_test_data,
                    max_encoder_timesteps=max_encoder_timesteps,
                    max_decoder_timesteps=max_decoder_timesteps,
                    max_transitions=max_sequence_length
                )

            else:
                # Non-sequence-to-sequence with matrix data using TCN.
                model.build_sequence_to_sequence_model_vector(
                    max_sequence_length=max_sequence_length,
                    no_of_input_features=no_of_input_features
                )
                model.train_vector_model_TCN(
                    train_data=shuffled_and_split_data['train'],
                    val_data=shuffled_and_split_data['validation'],
                    max_sequence_length=max_sequence_length
                )
                model_evaluator.evaluate_model(
                    model=model.model,
                    data=scaled_data,
                    min_max_values=min_max_values,
                    max_sequence_length=max_sequence_length,
                    test_cells=test_cells_id
                )

    # -----------------------------------------------------------------------------
    # Save Updated Configuration
    # -----------------------------------------------------------------------------
    # Insert min_max_values into the config.
    config['min_max_values'] = min_max_values

    # Convert bin_mapping to a serializable form.
    config["bin_mapping"] = make_serializable(data_processing.bin_mapping)

    # Write the updated configuration to the experiment result directory.
    with open(os.path.join(experiment_result_dir, 'config.json'), 'w') as json_file:
        json.dump(config, json_file, indent=4)

    print("Experiment Completed!")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
