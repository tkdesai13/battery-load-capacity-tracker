import os
import numpy as np
import tensorflow as tf
import random
import json
import warnings
from data.data_processing import DataProcessor
from models.model import Model
from plot.model_evaluation import ModelEvaluator


# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

# Load configuration from a JSON file
def load_config(config_path='config.json'):
    with open(config_path, 'r') as json_file:
        return json.load(json_file)

# Create a directory for experiment results
def create_experiment_dir(config):
    experiment_result_dir = os.path.join(config['paths']['result'], config['experiment_name'])
    if not os.path.exists(experiment_result_dir):
        os.makedirs(experiment_result_dir)
    return experiment_result_dir

def main():
    # Ignore  warnings
    warnings.filterwarnings("ignore")
    set_seeds()

    # Load configuration
    config = load_config()
    experiment_result_dir = create_experiment_dir(config)

    # Initialize data processing
    data_processing = DataProcessor(config=config)
    time_bins_and_transition_data = data_processing.load_and_binning_data()

    # Prepare features based on the experiment flag
    if config['scalar_vector_exp_flag'] == 'scalar':
        time_bins_and_transition_data, min_max_values = data_processing.prepare_scalar_features(
            time_bins_and_transition_data)
    elif config['scalar_vector_exp_flag'] == 'vector':
        time_bins_and_transition_data, min_max_values = data_processing.prepare_vector_features(
            time_bins_and_transition_data)
    else:
        time_bins_and_transition_data, min_max_values = data_processing.prepare_vector_features_embedded(
            time_bins_and_transition_data)

    # Scale data and split into training and testing sets
    scaled_data = data_processing.scale_data(time_bins_and_transition_data, min_max_values)
    shuffled_and_split_data, test_cells_id = data_processing.shuffle_and_split_data(scaled_data)

    # Initialize model and evaluator
    model = Model(config=config)
    model_evaluator = ModelEvaluator(config=config)

    # Load or build and train the model based on configuration
    if config['model_config']['trained_model'] == True and config['paths']['trained_model'] != '':
        max_sequence_length = int(min_max_values['max_sequence_length'])
        no_of_input_features = len(data_processing.bin_mapping)
        model.build_vector_model_TCN(max_sequence_length=max_sequence_length, no_of_input_features=no_of_input_features)
        model.load_model(exp_path=config['paths']['trained_model'])
        model_evaluator.evaluate_model(model=model.model, data=scaled_data, min_max_values=min_max_values,
                                       max_sequence_length=max_sequence_length, test_cells=test_cells_id)
    else:
        if config['scalar_vector_exp_flag'] == 'scalar':
            no_of_input_features = len(shuffled_and_split_data['train']['timebins'][0])
            model.build_scalar_model(no_of_input_features=no_of_input_features)
            model.train_model(train_data=shuffled_and_split_data['train'],
                              val_data=shuffled_and_split_data['validation'])
            model_evaluator.evaluate_model(model=model.model, data=scaled_data, min_max_values=min_max_values,
                                           test_cells=test_cells_id)
        elif config['scalar_vector_exp_flag'] == 'vector':
            max_sequence_length = int(min_max_values['max_transitions'])
            no_of_input_features = len(data_processing.bin_mapping)
            model.build_vector_model(max_sequence_length=max_sequence_length, no_of_input_features=no_of_input_features)
            model.train_vector_model(train_data=shuffled_and_split_data['train'],
                                     val_data=shuffled_and_split_data['validation'],
                                     max_sequence_length=max_sequence_length)
            model_evaluator.evaluate_model(model=model.model, data=scaled_data, min_max_values=min_max_values,
                                           max_sequence_length=max_sequence_length, test_cells=test_cells_id)
        else:
            max_sequence_length = int(min_max_values['max_sequence_length'])
            no_of_input_features = len(data_processing.bin_mapping)
            model.build_vector_model_embedded_convo(max_sequence_length=max_sequence_length, no_of_input_features=no_of_input_features)
            model.train_vector_model_TCN(train_data=shuffled_and_split_data['train'],
                                     val_data=shuffled_and_split_data['validation'],
                                     max_sequence_length=max_sequence_length)
            model_evaluator.evaluate_model(model=model.model, data=scaled_data, min_max_values=min_max_values,
                                           max_sequence_length=max_sequence_length, test_cells=test_cells_id)

    # Save updated configuration with min_max_values
    config['min_max_values'] = min_max_values
    with open(os.path.join(experiment_result_dir, 'config.json'), 'w') as json_file:
        json.dump(config, json_file, indent=4)

    # Print bin mapping and intervals
    bin_mapping = data_processing.get_bin_mapping()
    print("Bin mapping:", bin_mapping)
    v_bin, c_bin, t_bin = bin_mapping[0]
    intervals = data_processing.get_bin_intervals(v_bin, c_bin, t_bin)
    print(f"Voltage Interval: {intervals[0]}, Current Interval: {intervals[1]}, Temperature Interval: {intervals[2]}")
    print("Experiment Completed!")


if __name__ == "__main__":
    main()
