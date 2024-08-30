import os
import numpy as np
import tensorflow as tf
import random
import json
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

# Initialize the DataProcessor class
data_processing = DataProcessor(config=config)

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

# Function to create plots for each cell and accumulate them
def create_plots(data_processing, output_dir):
    plots = []
    cell_categories = {'train_cells': 'Training', 'test_cells': 'Testing'}

    for cell_category, cell_label in cell_categories.items():
        for cell_name in data_processing.config['cells'][cell_category]:
            print(f"Processing cell: {cell_name}")

            # Load data for the specific cell
            file_path = data_processing._construct_file_path(data_processing.config['paths']['data'], cell_name)
            raw_cell_data = data_processing._load_and_process_file(file_path)
            cell_data = data_processing._accumulate_cell_data(raw_cell_data)

            voltage = cell_data['voltage']
            current = cell_data['current']
            temperature = cell_data['temperature']

            # Create probability histogram plots
            p1 = probability_histogram(voltage, f"{cell_label} Cell {cell_name} - Voltage Distribution",
                                       x_label='Voltage (V)', y_label='Probability', color="navy")

            p2 = probability_histogram(current, f"{cell_label} Cell {cell_name} - Current Distribution",
                                       x_label='Current (A)', y_label='Probability', color="green")

            p3 = probability_histogram(temperature, f"{cell_label} Cell {cell_name} - Temperature Distribution",
                                       x_label='Temperature (°C)', y_label='Probability', color="red")

            # Arrange plots in a row layout
            plots.append(row(p1, p2, p3))

    # Save all plots to a single HTML file
    output_filename = os.path.join(output_dir, "cell_data_plots.html")
    output_file(output_filename)
    layout = column(*plots)
    save(layout)

# Call the function to create and save the plots in the experiment results directory
create_plots(data_processing, experiment_result_dir)
