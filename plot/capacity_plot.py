import os
import json
import random
import logging
import warnings

# Set TF logging level environment variable
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data.data_processing import DataProcessor
from models.model import Model
from plot.model_evaluation import ModelEvaluator
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import ColumnDataSource, HoverTool

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Lucida Bright'],
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage{mathpazo} \usepackage[T1]{fontenc}'
})


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


def load_config(config_path: str = '../config.json') -> dict:
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
    # Set seeds and load configuration
    set_seeds()
    config = load_config()
    experiment_result_dir = create_experiment_dir(config)

    # Load data
    data_processing = DataProcessor(config=config)
    time_bins_and_transition_data = data_processing.load_and_binning_data()

    # Define groups:
    # Severson dataset: batches 'batch1', 'batch2', 'batch3'
    # Attia dataset: batch 'b9'
    severson_batches = ['batch1', 'batch2', 'batch3']
    attia_batch = 'b9'

    # Create a single figure
    fig, ax = plt.subplots(figsize=(20, 6))

    # Define colors for the two groups
    severson_color = 'black'
    attia_color = 'blue'

    # Plot individual cell curves for the Severson group
    first_severson = True  # For legend label only once
    for batch in severson_batches:
        batch_data = time_bins_and_transition_data[batch]
        for cell_id, cell_data in batch_data.items():
            cycles = []
            capacities = []
            # Ensure cycles are sorted numerically
            for cycle_key in sorted(cell_data.keys(), key=lambda x: int(x)):
                gt = cell_data[cycle_key].get('ground_truth_cycle', {})
                cycle_no = gt.get('cycle_no', None)
                capacity = gt.get('capacity', None)
                if cycle_no is not None and capacity is not None:
                    cycles.append(cycle_no)
                    capacities.append(capacity)
            if cycles and capacities:
                if first_severson:
                    ax.plot(cycles, capacities, color=severson_color,
                            linestyle='-', alpha=0.7, linewidth=1, label='Severson')
                    first_severson = False
                else:
                    ax.plot(cycles, capacities, color=severson_color,
                            linestyle='-', alpha=0.7, linewidth=1)

    # Plot individual cell curves for the Attia group
    first_attia = True  # For legend label only once
    batch_data = time_bins_and_transition_data[attia_batch]
    for cell_id, cell_data in batch_data.items():
        cycles = []
        capacities = []
        for cycle_key in sorted(cell_data.keys(), key=lambda x: int(x)):
            gt = cell_data[cycle_key].get('ground_truth_cycle', {})
            cycle_no = gt.get('cycle_no', None)
            capacity = gt.get('capacity', None)
            if cycle_no is not None and capacity is not None:
                cycles.append(cycle_no)
                capacities.append(capacity)
        if cycles and capacities:
            if first_attia:
                ax.plot(cycles, capacities, color=attia_color,
                        linestyle='-', alpha=0.7, linewidth=1, label='Attia')
                first_attia = False
            else:
                ax.plot(cycles, capacities, color=attia_color,
                        linestyle='-', alpha=0.7, linewidth=1)

    # Axis labels, legend, and grid
    ax.set_xlabel('Cycle Number', fontsize=23)
    ax.set_ylabel('Capacity (Ah)', fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=20)
    # ax.grid(True)
    ax.legend(fontsize=21)

    # Set x-axis to start from cycle number 1
    ax.set_xlim(left=1)

    plt.tight_layout()
    # Save the figure as a high-resolution PDF file
    # pdf_path = os.path.join('cycle_vs_capacity.pdf')
    plt.savefig('cycle_vs_capacity.pdf', format='pdf', dpi=300)
    plt.show()



    # # bokeh plot
    # # Create a Bokeh figure
    # p = figure(width=1200, height=600,
    #            title="Cycle Number vs Capacity",
    #            x_axis_label="Cycle Number", y_axis_label="Capacity (Ah)")
    # # Add hover tool with tooltips
    # hover = HoverTool(tooltips=[("Cell", "@cell"), ("Cycle", "@x"), ("Capacity", "@y")])
    # p.add_tools(hover)
    #
    # # Plot each cell from Severson group
    # for batch in severson_batches:
    #     batch_data = time_bins_and_transition_data[batch]
    #     for cell_id, cell_data in batch_data.items():
    #         cycles = []
    #         capacities = []
    #         # Sort cycles numerically
    #         for cycle_key in sorted(cell_data.keys(), key=lambda x: int(x)):
    #             gt = cell_data[cycle_key].get('ground_truth_cycle', {})
    #             cycle_no = gt.get('cycle_no')
    #             capacity = gt.get('capacity')
    #             if cycle_no is not None and capacity is not None:
    #                 cycles.append(cycle_no)
    #                 capacities.append(capacity)
    #         if cycles and capacities:
    #             source = ColumnDataSource(data=dict(x=cycles, y=capacities, cell=[cell_id] * len(cycles)))
    #             p.line('x', 'y', source=source, line_width=2, color=severson_color)
    #
    # # Plot each cell from Attia group
    # batch_data = time_bins_and_transition_data[attia_batch]
    # for cell_id, cell_data in batch_data.items():
    #     cycles = []
    #     capacities = []
    #     for cycle_key in sorted(cell_data.keys(), key=lambda x: int(x)):
    #         gt = cell_data[cycle_key].get('ground_truth_cycle', {})
    #         cycle_no = gt.get('cycle_no')
    #         capacity = gt.get('capacity')
    #         if cycle_no is not None and capacity is not None:
    #             cycles.append(cycle_no)
    #             capacities.append(capacity)
    #     if cycles and capacities:
    #         source = ColumnDataSource(data=dict(x=cycles, y=capacities, cell=[cell_id] * len(cycles)))
    #         p.line('x', 'y', source=source, line_width=2, color=attia_color)
    #
    # # Set x-axis to start at cycle number 1
    # p.x_range.start = 1
    #
    # # Output to an HTML file and display
    # output_file("cycle_vs_capacity_interactive.html")
    # save(p)


# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
