# Description: This file is the main entry point for the project.
# It reads the configuration file, extracts data, and processes it.

import os
import json
import warnings
from data.data_processing import DataProcessor
from models.model import Model
from plot.model_evaluation import Plot

# Ignore warnings
warnings.filterwarnings("ignore")

# Open the JSON Configuration file
with open('config.json', 'r') as json_file:
    config = json.load(json_file)

# Construct the path for experiment-specific results
experiment_result_dir = os.path.join(config['paths']['result'], config['experiment_name'])

# Ensure the directory exists, and create it if it doesn't
if not os.path.exists(experiment_result_dir):
    os.makedirs(experiment_result_dir)

# data Extraction for Model training
print("Start: Data extraction and Processing")

# Initialize the DataProcessor class
data_processing = DataProcessor(config=config)

# Load and preprocess the data cell by cell
data = data_processing.load_and_preprocess_data()

# prepare the data for training and testing
train_test_data = data_processing.prepare_features_target_variables(data)

# shuffle and split the data
shuffled_and_split_data = data_processing.shuffle_and_split_data(train_test_data)

# Initialize the Model class
model = Model(config=config)

# Build the model or load the trained model
if config['model_config']['trained_model'] == 'True' and config['paths']['trained_model'] != '':
    model.load_model(exp_path=config['paths']['trained_model'])
else:
    input_features_size = len(shuffled_and_split_data['train_cells']['features'][0])
    model.build_model(input_features_size=input_features_size)

    # Train the model
    model.train_model(train_data=shuffled_and_split_data['train_cells'],
                      val_data=shuffled_and_split_data['validation_cells'])

# plot the results
plot = Plot(config=config)
plot.evaluate_model(model=model.model, test_data=shuffled_and_split_data['test_cells'])


