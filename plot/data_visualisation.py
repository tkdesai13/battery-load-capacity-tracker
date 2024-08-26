# Description: This file is the main entry point for the project.
# It reads the configuration file, extracts data, and processes it.

import os
import numpy as np
import tensorflow as tf
import random
import json
import warnings
from data.data_processing import DataProcessor
from models.model import Model
from plot.model_evaluation import ModelEvaluator

# reproducibility
# Set seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Ignore warnings
warnings.filterwarnings("ignore")

# Open the JSON Configuration file
with open('../config.json', 'r') as json_file:
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
data = data_processing.load_and_acccumulate_data()