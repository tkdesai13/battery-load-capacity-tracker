# Description: This file is the main entry point for the project.
# It reads the configuration file, extracts data, and processes it.

import os
import numpy as np
import torch
import random
import json
import warnings
from data.data_processing import DataProcessor
from models.gnn_model import GNNModel
from plot.model_evaluation import ModelEvaluator
from torch_geometric.loader import DataLoader

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

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

# Data Extraction for Model training
print("Start: Data extraction and Processing")

# Initialize the DataProcessor class
data_processing = DataProcessor(config=config)

# Load and preprocess the data cell by cell
data = data_processing.load_and_binning_data()

# Prepare graph for training and testing
graph_data = data_processing.prepare_graph_data(data)

# Split the graph data into train, validation
train_graphs, val_graphs = data_processing.shuffle_and_split_graph_data(graph_data)

# Create DataLoaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)

# Test cells remain the same
test_graphs = [graph for cycles in graph_data['test_cells'].values() for graph in cycles.values()]
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# Initialize the GNN Model class
gnn_model = GNNModel(config=config)

# Build the model or load the trained model
if config['model_config']['trained_model'] == 'True' and config['paths']['trained_model'] != '':
    gnn_model.load_model(exp_path=config['paths']['trained_model'])
else:
    # build the model
    gnn_model.build_model()
    # Train the model
    gnn_model.train_model(train_loader=train_loader, val_loader=val_loader)

# Evaluate the model
model_evaluator = ModelEvaluator(config=config)
model_evaluator.evaluate_model(gnn_model, test_loader)

# Save the configuration file
with open(os.path.join(experiment_result_dir, 'config.json'), 'w') as json_file:
    json.dump(config, json_file, indent=4)

