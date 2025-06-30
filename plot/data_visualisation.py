import os
import numpy as np
import tensorflow as tf
import random
import json
import pickle
import warnings
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.models import HoverTool
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from bokeh.models import ColorBar, LinearColorMapper
from bokeh.transform import linear_cmap
from scipy.stats import gaussian_kde
from bokeh.palettes import Viridis256
from itertools import chain

from data.data_processing import DataProcessor

# Reproducibility
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

# Data Extraction for Model training
print("Start: Data extraction and Processing")


# Function to calculate and plot a normalized histogram (probability histogram)
def probability_histogram(data, title, x_label, y_label, color):
    # Normalized histogram
    hist, edges = np.histogram(data, bins=50, density=True)
    hist_centers = (edges[:-1] + edges[1:]) / 2

    # Calculate the width of each bin
    bin_width = edges[1] - edges[0]

    # Probability for each bin (height multiplied by bin width should sum to 1)
    probabilities = hist * bin_width

    p = figure(title=title, x_axis_label=x_label, y_axis_label=y_label)

    # Add the probability histogram to the plot
    p.vbar(x=hist_centers, top=probabilities, width=bin_width, fill_alpha=0.6, color=color)

    return p

# Function to flatten list of lists into a single list
def flatten(data):
    return list(chain.from_iterable(data))

# Function to deeply flatten NumPy arrays and nested lists
def deep_flatten(data):
    """
    Flattens a list of NumPy arrays or nested lists into a single list.
    """
    return list(chain.from_iterable([np.asarray(item).flatten().tolist() if isinstance(item, np.ndarray) else item for item in data]))

# Updated function to create and save plots for each batch
def create_batch_plots(output_dir):
    data_path = config['paths']['data']

    # Ensure the data path exists
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"The data path {data_path} does not exist.")

    # Iterate over all .pkl files in the data directory
    for file_name in os.listdir(data_path):
        if not file_name.endswith('.pkl'):
            continue  # Skip non-pickle files

        print(f"Processing file: {file_name}")
        file_path = os.path.join(data_path, file_name)
        try:
            with open(file_path, 'rb') as f:
                batch_data = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            continue  # Skip to the next file if loading fails

        plots = []  # Store plots for all cells in this batch

        # Iterate through each cell in the batch
        for cell_name, cell_data in batch_data.items():
            print(f"Processing cell: {cell_name}")

            # Extract cycle data (excluding the first cycle)
            cycles = cell_data.get('cycles', {})
            if not cycles:
                print(f"No cycle data found for cell: {cell_name}")
                continue

            cycle_keys = sorted(cycles.keys(), key=int)  # Ensure keys are sorted numerically
            if len(cycle_keys) > 1:  # Ensure there is more than one cycle
                cycles_to_use = {k: cycles[k] for k in cycle_keys[1:]}  # Exclude the first cycle
            else:
                print(f"Skipping cell {cell_name} since it has only one cycle.")
                continue

            # Properly flatten NumPy arrays & nested lists
            voltage_data = deep_flatten([cycle_data.get('V', []) for cycle_data in cycles_to_use.values()])
            current_data = deep_flatten([cycle_data.get('I', []) for cycle_data in cycles_to_use.values()])
            temperature_data = deep_flatten([cycle_data.get('T', []) for cycle_data in cycles_to_use.values()])

            # Create probability histogram plots
            p1 = probability_histogram(
                voltage_data,
                title=f"Cell {cell_name} - Voltage Distribution (Excluding First Cycle)",
                x_label="Voltage (V)",
                y_label="Probability",
                color="navy"
            )
            p2 = probability_histogram(
                current_data,
                title=f"Cell {cell_name} - Current Distribution (Excluding First Cycle)",
                x_label="Current (A)",
                y_label="Probability",
                color="green"
            )
            p3 = probability_histogram(
                temperature_data,
                title=f"Cell {cell_name} - Temperature Distribution (Excluding First Cycle)",
                x_label="Temperature (°C)",
                y_label="Probability",
                color="red"
            )

            # Arrange the three plots for this cell in a row layout
            plots.append(row(p1, p2, p3))

        # Save all plots for this batch to a single HTML file
        batch_id = file_name.split('.')[0]
        output_filename = os.path.join(output_dir, f"{batch_id}_cell_data_plots.html")
        output_file(output_filename)
        layout = column(*plots)
        save(layout)
        print(f"Plots for batch {batch_id} saved to {output_filename}")


# Call the function to create and save the plots in the experiment results directory
if __name__ == '__main__':
    create_batch_plots(experiment_result_dir)
