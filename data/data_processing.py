# Description: This file contains the DataProcessor class, which is responsible for loading, preprocessing, and
# structuring the data for training and testing the model.

import os
import itertools
from datetime import datetime
import scipy.io
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data


def matlab_struct_to_dict(matlab_struct: Any) -> dict:
    """
    Recursively converts a MATLAB struct (loaded as mat_struct) to a Python dictionary.

    :param matlab_struct: The MATLAB struct to convert.
    :return: The converted Python dictionary.
    """
    result = {}
    for field_name in matlab_struct._fieldnames:
        field_value = getattr(matlab_struct, field_name)
        if isinstance(field_value, scipy.io.matlab.mio5_params.mat_struct):
            result[field_name] = matlab_struct_to_dict(field_value)
        elif isinstance(field_value, np.ndarray):
            result[field_name] = matlab_struct_array_to_list(field_value)
        else:
            result[field_name] = field_value
    return result


def matlab_struct_array_to_list(ndarray: np.ndarray) -> list:
    """
    Converts a MATLAB struct array into a list of Python dictionaries or returns the ndarray
    as a list if it is not a struct.

    :param ndarray: The numpy array to convert.
    :return: The converted list.
    """
    if ndarray.dtype == 'O':  # object type, usually for structs
        return [matlab_struct_to_dict(elem) for elem in ndarray]
    return ndarray.tolist()


class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.voltage_bins, self.current_bins, self.temperature_bins, self.num_bins_per_dim = self.initialize_bins()
        self.bin_mapping = self.create_bin_mapping()  # Initialize bin mapping for tuple-based indexing
        self.transitions_per_bin_pair = self.initialize_transitions_per_bin_pair()

        # Initialize global min/max trackers for all bins and transitions
        self.global_bin_min_max_values = {
            'min_time_spent': {bin_index: float('inf') for bin_index in self.bin_mapping.keys()},
            'max_time_spent': {bin_index: float('-inf') for bin_index in self.bin_mapping.keys()},
            'min_transitions': {bin_pair: float('inf') for bin_pair in self.transitions_per_bin_pair.keys()},
            'max_transitions': {bin_pair: float('-inf') for bin_pair in self.transitions_per_bin_pair.keys()},
        }

        # Initialize global min/max trackers for capacity-related values
        self.global_capacity_min_max_values = {
            'min_capacity_diff': float('inf'),
            'max_capacity_diff': float('-inf'),
            'min_previous_capacity': float('inf'),
            'max_previous_capacity': float('-inf'),
            'min_current_capacity': float('inf'),
            'max_current_capacity': float('-inf'),
        }

    def initialize_bins(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Initialize the bins for voltage, current, and temperature based on the configuration.
        """
        voltage_bins = np.array(self.config['time_bins']['voltage_bins'])
        current_bins = np.array(self.config['time_bins']['current_bins'])
        temperature_bins = np.array(self.config['time_bins']['temperature_bins'])

        num_bins_per_dim = [len(voltage_bins), len(current_bins), len(temperature_bins)]

        return voltage_bins, current_bins, temperature_bins, num_bins_per_dim

    def create_bin_mapping(self) -> dict:
        """
        Create a mapping from bin index tuples to voltage, current, and temperature intervals.
        """
        bin_mapping = {}
        for i, j, k in itertools.product(range(len(self.voltage_bins)),
                                         range(len(self.current_bins)),
                                         range(len(self.temperature_bins))):
            bin_mapping[(i, j, k)] = {
                'voltage_interval': self.voltage_bins[i],
                'current_interval': self.current_bins[j],
                'temperature_interval': self.temperature_bins[k]
            }
        return bin_mapping

    def initialize_transitions_per_bin_pair(self):
        """
        Initialize all possible transitions between bins, excluding self-transitions.
        """
        transitions_per_bin_pair = defaultdict(int)
        bin_indices = list(self.bin_mapping.keys())
        for start_bin in bin_indices:
            for end_bin in bin_indices:
                if start_bin != end_bin:  # Exclude self-pairs
                    transitions_per_bin_pair[(start_bin, end_bin)] = 0
        return transitions_per_bin_pair

    def load_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the data from the specified path and structure it into a dictionary.
        """
        data_path = self.config['paths']['data']
        raw_data = {}

        for cell_type, cells in self.config['cells'].items():
            raw_data[cell_type] = {}
            for cell in cells:
                cell_name = str(cell)
                file_path = self.construct_file_path(data_path, cell_name)
                raw_data[cell_type][cell_name] = self.load_mat_file(file_path)

        return raw_data

    def _construct_file_path(self, base_path: str, cell_name: str) -> str:
        """
        Construct the file path for the .mat file.

        :param base_path: The base path for the data.
        :param cell_name: The name of the cell.
        :return: The full file path.
        """
        return os.path.join(base_path, f"{cell_name}.mat")

    def _matlab_struct_to_dict(self, file_path: str) -> Dict[str, Any]:
        """
        Load and process the .mat file, converting it to a dictionary.

        :param file_path: The path to the .mat file.
        :return: The processed data as a dictionary.
        """
        try:
            mat_data = scipy.io.loadmat(file_name=file_path, squeeze_me=True, struct_as_record=False)
            return matlab_struct_to_dict(mat_data.get('data', {}))
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None

    def load_and_binning_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the data from the data path, process, and structure it into a dictionary.

        :return: The preprocessed data structured as a dictionary.
        """
        data_path = self.config['paths']['data']
        preprocessed_data = {}

        for cell_type, cells in self.config['cells'].items():
            preprocessed_data[cell_type] = {}
            for cell in cells:
                cell_name = str(cell)
                print(f"Loading data for {cell_type}/{cell_name}")
                file_path = self._construct_file_path(data_path, cell_name)
                raw_cell_data = self._matlab_struct_to_dict(file_path)
                preprocessed_data[cell_type][cell_name] = self._process_cell_data(raw_cell_data)

        return preprocessed_data

    def _process_cell_data(self, cell_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """
        Preprocess the data for a single cell.
        """
        cycle_counter = 1
        processed_data = {}
        accumulated_data = self._initialize_accumulated_data()

        for step_data in cell_data['step']:
            self._accumulate_step_data(accumulated_data, step_data)

            if step_data['comment'] == 'reference discharge':
                if cycle_counter > 1 and not self._is_valid_capacity_change(processed_data, cycle_counter, step_data):
                    continue

                processed_load_pattern = self._process_load_pattern(accumulated_data)

                processed_data[cycle_counter] = {
                    'accumulated_time_bins': processed_load_pattern["time_bins"],
                    'accumulated_transitions': processed_load_pattern["transitions"],
                    'ground_truth_cycle': self._process_reference_capacity(step_data)
                }

                accumulated_data = self._initialize_accumulated_data()
                cycle_counter += 1

        return processed_data

    def _initialize_accumulated_data(self) -> dict:
        """
        Initialize the structure for accumulating data.
        """
        return {
            "voltage": [],
            "current": [],
            "temperature": [],
            "time": []
        }

    def _accumulate_step_data(self, accumulated_data: dict, step_data: dict):
        """
        Accumulate data from a single step into the accumulated data structure.
        """
        for key in accumulated_data.keys():
            value = step_data[key]
            if isinstance(value, list):
                accumulated_data[key].extend(value)
            else:
                accumulated_data[key].append(value)

    def _is_valid_capacity_change(self, processed_data: dict, cycle_counter: int, step_data: dict) -> bool:
        """
        Validate the capacity change between cycles.
        """
        current_capacity = self._process_reference_capacity(step_data)
        previous_capacity = processed_data[cycle_counter - 1]['ground_truth_cycle']['capacity']
        capacity_diff = abs(current_capacity['capacity'] - previous_capacity)
        return 0.01 <= capacity_diff <= 0.4

    def _process_load_pattern(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the load pattern data by calculating the time spent in each bin and transitions between bins.
        """
        voltage_data = step_data["voltage"]
        current_data = step_data["current"]
        temperature_data = step_data["temperature"]
        time_data = step_data["time"]

        time_bins, transitions = self._bin_data_vector(voltage_data, current_data, temperature_data, time_data)
        return {"time_bins": time_bins, "transitions": transitions}

    def _bin_data_vector(self, voltage, current, temperature, time) -> Tuple[list, list]:
        """
        Bin the battery data and calculate the vectorized time spent in each bin and transitions between bins.
        """
        v_indices = self._digitize_data(voltage, self.voltage_bins)
        c_indices = self._digitize_data(current, self.current_bins)
        t_indices = self._digitize_data(temperature, self.temperature_bins)

        combined_indices = np.stack((v_indices, c_indices, t_indices), axis=-1)
        time_diffs = np.diff(time) / 3600  # Convert time differences to hours

        transitions = np.where(np.any(combined_indices[1:] != combined_indices[:-1], axis=1))[0]

        time_bins_dict = []
        transition_dict = []

        bin_start = 0
        for transition_idx in transitions:
            time_spent = np.sum(time_diffs[bin_start:transition_idx + 1])
            bin_tuple = tuple(combined_indices[bin_start])

            time_bins_dict.append({
                'bin_index': bin_tuple,
                'time_spent': time_spent,
                'transition_index': bin_start
            })

            transition_dict.append({
                'from_bin': bin_tuple,
                'to_bin': tuple(combined_indices[transition_idx + 1]),
                'transition_index': transition_idx + 1
            })

            bin_start = transition_idx + 1

        if bin_start < len(combined_indices) - 1:
            time_spent = np.sum(time_diffs[bin_start:])
            time_bins_dict.append({
                'bin_index': tuple(combined_indices[bin_start]),
                'time_spent': time_spent,
                'transition_index': bin_start
            })

        return time_bins_dict, transition_dict

    def _digitize_data(self, data, bins) -> np.ndarray:
        """
        Digitize data into bin indices.
        """
        indices = np.digitize(data, bins[:, 1], right=True)
        return np.clip(indices, 0, len(bins) - 1)

    def _process_reference_capacity(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the capacity measurement data.
        """
        date_str = step_data.get("date", "")
        step_date = datetime.strptime(date_str, "%d-%b-%Y %H:%M:%S") if date_str else None

        current_data = step_data["current"]
        relative_time_data = step_data["relativeTime"]
        relative_time_hours = np.array(relative_time_data) / 3600.0

        capacity = np.trapz(current_data, relative_time_hours)

        return {"date": step_date, "capacity": capacity}

    def prepare_scalar_features(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict:
        all_cell_target_feature_data = {}

        for cell_type, cells in preprocessed_data.items():
            all_cell_target_feature_data[cell_type] = {}
            for cell_name, cell_data in cells.items():
                print(f"Preparing scalar features for {cell_type}/{cell_name}")
                all_cell_target_feature_data[cell_type][cell_name] = {}

                previous_capacity = self.config['nominal_capacity']  # Start with nominal capacity

                for cycle_no, cycle_data in sorted(cell_data.items()):
                    time_bins = cycle_data.get('accumulated_time_bins', [])
                    transitions = cycle_data.get('accumulated_transitions', [])
                    current_capacity = cycle_data['ground_truth_cycle'].get('capacity', previous_capacity)

                    # Initialize cycle-specific data structures with zero values for all bins and bin pairs
                    cycle_accumulated_time_bins = {key: 0.0 for key in self.bin_mapping.keys()}
                    cycle_transition_counts = self.initialize_transitions_per_bin_pair()

                    # Accumulate time for each bin across all transitions within a cycle
                    for entry in time_bins:
                        bin_index = entry['bin_index']
                        time_spent = entry['time_spent']
                        cycle_accumulated_time_bins[bin_index] += time_spent

                    # Accumulate transitions between bins
                    for transition in transitions:
                        start_bin = transition['from_bin']
                        end_bin = transition['to_bin']
                        cycle_transition_counts[(start_bin, end_bin)] += 1

                    # Calculate capacity difference
                    capacity_diff = current_capacity - previous_capacity

                    # Store the features and target in the output dictionary
                    all_cell_target_feature_data[cell_type][cell_name][cycle_no] = {
                        'accumulated_time_bins': cycle_accumulated_time_bins,
                        'accumulated_transitions': cycle_transition_counts,
                        'capacity': current_capacity,
                        'date': cycle_data['ground_truth_cycle']['date'],
                        'capacity_diff': capacity_diff,
                        'previous_capacity': previous_capacity
                    }

                    # Update previous capacity for next cycle
                    previous_capacity = current_capacity

        return all_cell_target_feature_data

    def prepare_vector_features(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict:
        """
        Prepare the data for training and testing by converting the scalar features to vectorized features.

        :param preprocessed_data: The preprocessed data structured as a dictionary.
        :return: The preprocessed data with vector features.
        """
        for cell_type, cells in preprocessed_data.items():
            for cell_name, cell_data in cells.items():
                print(f"Preparing vector features for {cell_type}/{cell_name}")

                for cycle_no, cycle_data in cell_data.items():
                    time_bins = cycle_data.get('accumulated_time_bins', {})
                    transition_matrix = cycle_data.get('accumulated_transitions', {})

                    # Convert the scalar time bins to vectorized features
                    vector_time_bins = self.generate_transition_matrix(time_bins)

                    # Update the cycle data with vector features
                    cell_data[cycle_no]['accumulated_time_bins'] = vector_time_bins
                    cell_data[cycle_no]['accumulated_transitions'] = transition_matrix

        return preprocessed_data

    def generate_transition_matrix(self, time_bins: dict) -> np.ndarray:
        """
        Generate a matrix of shape (no_of_transitions, no_of_bins) where each row corresponds to a transition
        and only the specific bin column has the time spent.

        :param time_bins: A list of dictionaries with keys 'bin_index', 'time_spent', 'transition_index'.
        :return: A numpy matrix of shape (no_of_transitions, no_of_bins).
        """
        no_of_bins = self.num_bins_per_dim[0] * self.num_bins_per_dim[1] * self.num_bins_per_dim[2]
        no_of_transitions = len(time_bins)
        matrix = np.zeros((no_of_transitions, no_of_bins))

        for i, transition in enumerate(time_bins):
            bin_index = transition['bin_index']
            time_spent = transition['time_spent']
            matrix[i, bin_index] = time_spent

        return matrix

    def _add_time_bins(self, accumulated_time_bins: dict, current_time_bins: dict) -> dict:
        """
        Add the time spent in the current step's bins to the accumulated time bins.

        :param accumulated_time_bins: The accumulated time bins.
        :param current_time_bins: The time bins for the current step.
        :return: The updated accumulated time bins.
        """
        for key, value in current_time_bins.items():
            accumulated_time_bins[key] += value
        return accumulated_time_bins

    def _update_transition_matrix(self, accumulated_transitions: dict, current_transitions: dict) -> dict:
        """
        Update the accumulated transition matrix with the current step's transitions.

        :param accumulated_transitions: The accumulated transition matrix.
        :param current_transitions: The transition matrix for the current step.
        :return: The updated accumulated transition matrix.
        """
        for key, value in current_transitions.items():
            accumulated_transitions[key] += value
        return accumulated_transitions

    def prepare_features_target_variables(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict:
        """
        Prepare the data for training and testing.

        :param preprocessed_data: The preprocessed data structured as a dictionary.
        :return: A dictionary containing the training and testing data, with features and delta_Q as the target variables.
        """

        train_test_data = {}

        for cell_type, cells in preprocessed_data.items():
            train_test_data[cell_type] = {}

            for cell_name, cell_data in cells.items():
                print(f"Preparing features and target variable for {cell_type}/{cell_name}")

                # Initialize the dictionary for this cell_name
                train_test_data[cell_type][cell_name] = {"features": [], "target": [], "ground_truth_cycle": []}

                previous_capacity = self.config['nominal_capacity']

                for cycle_no, cycle_data in cell_data.items():
                    # Prepare features: accumulated time bins + previous cycle's capacity
                    accumulated_time_bins = list(cycle_data.get('accumulated_time_bins', {}).values())
                    features = accumulated_time_bins + [previous_capacity]

                    # Prepare target: delta_Q
                    current_capacity = cycle_data.get('ground_truth_cycle', {}).get('capacity', previous_capacity)
                    current_capacity_date = cycle_data.get('ground_truth_cycle', {}).get('date', None)
                    delta_q = current_capacity - previous_capacity

                    # Store the features and target in the dictionary
                    train_test_data[cell_type][cell_name]["features"].append(features)
                    train_test_data[cell_type][cell_name]["target"].append(delta_q)

                    # Store the capacity and date together in ground_truth_cycle
                    train_test_data[cell_type][cell_name]["ground_truth_cycle"].append({
                        "capacity": current_capacity,
                        "date": current_capacity_date
                    })

                    # Update previous_capacity for the next cycle
                    previous_capacity = current_capacity

        # Returning the structured data for training and testing
        return train_test_data

    def shuffle_and_split_data(self, train_test_data: Dict[str, Dict[str, Dict[str, Any]]]) -> (
            Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]):
        """
        Shuffle the training data by combining all training cells together,
        and return the shuffled training data, validation data, and unaltered testing data.

        :param train_test_data: The training and testing data structured by cell type and cell name.
        :return: A tuple containing the training data, validation data, and unshuffled testing data.
        """

        split_data = {
            "train_cells": {
                "features": [],
                "target": [],
                "ground_truth_cycle": []
            },
            "validation_cells": {
                "features": [],
                "target": [],
                "ground_truth_cycle": []
            },
            "test_cells": {}
        }

        # Combine and shuffle training data
        combined_features = []
        combined_target = []
        combined_ground_truth_capacity = []

        for cell_name, data in train_test_data["train_cells"].items():
            combined_features.extend(data["features"])
            combined_target.extend(data["target"])
            combined_ground_truth_capacity.extend(data["ground_truth_cycle"])

        # Shuffle the combined features and targets together
        combined = list(zip(combined_features, combined_target, combined_ground_truth_capacity))
        np.random.shuffle(combined)
        shuffled_features, shuffled_target, shuffled_ground_truth_capacity = zip(*combined)

        # Calculate the split index for validation
        val_split_ratio = self.config['model_specifications']['val_train_split_ratio']
        split_index = int(len(shuffled_features) * (1 - val_split_ratio))

        # Split the data into training and validation sets
        split_data["train_cells"]["features"] = list(shuffled_features[:split_index])
        split_data["train_cells"]["target"] = list(shuffled_target[:split_index])
        split_data["train_cells"]["ground_truth_cycle"] = list(shuffled_ground_truth_capacity[:split_index])
        split_data["validation_cells"]["features"] = list(shuffled_features[split_index:])
        split_data["validation_cells"]["target"] = list(shuffled_target[split_index:])
        split_data["validation_cells"]["ground_truth_cycle"] = list(shuffled_ground_truth_capacity[split_index:])

        # Retain test data without shuffling
        split_data["test_cells"] = train_test_data["test_cells"]

        return split_data

    def shuffle_and_split_graph_data(self, graph_data):
        """
        :param graph_data:  The graph data structured by cell type, cell name, and cycle number.
        :return:
        """

        train_graphs = []
        val_graphs = []
        val_split_ratio = self.config['model_specifications']['val_train_split_ratio']

        for cell_name, cycles in graph_data['train_cells'].items():
            cycles_list = list(cycles.items())  # Convert to a list of tuples (cycle_no, graph)
            np.random.shuffle(cycles_list)

            split_index = int(len(cycles_list) * (1 - val_split_ratio))
            train_cycles = cycles_list[:split_index]
            val_cycles = cycles_list[split_index:]

            # Append the graphs to their respective lists
            for _, graph in train_cycles:
                train_graphs.append(graph)
            for _, graph in val_cycles:
                val_graphs.append(graph)

        return train_graphs, val_graphs

    def load_and_acccumulate_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the data from the data path and structure it into a dictionary.

        :return: The raw data from all cycles structured as a dictionary.
        """
        data_path = self.config['paths']['data']
        raw_data = {}

        for cell_type, cells in self.config['cells'].items():
            raw_data[cell_type] = {}
            for cell in cells:
                cell_name = str(cell)
                print(f"Loading data for {cell_type}/{cell_name}")
                file_path = self._construct_file_path(data_path, cell_name)
                raw_cell_data = self._matlab_struct_to_dict(file_path)
                raw_data[cell_type][cell_name] = self._accumulate_cell_data(raw_cell_data)

        return raw_data

    def _accumulate_cell_data(self, cell_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for a single cell.

        :param cell_data: The raw data for a single cell.
        :return: The accumulated data for the cell.
        """

        # Accumulated data dictionaries
        accumulated_data = {
            "voltage": [],
            "current": [],
            "temperature": [],
            "time": []
        }

        for step_counter, step_data in enumerate(cell_data['step']):

            # Accumulate data for each key
            for key in accumulated_data.keys():
                value = step_data[key]
                if isinstance(value, list):
                    accumulated_data[key].extend(value)
                else:
                    accumulated_data[key].append(value)

        return accumulated_data

    def prepare_graph_data(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Prepare the data for training and testing by constructing graph data for each cycle.

        :param preprocessed_data: The preprocessed data structured as a dictionary.
        :return: A dictionary containing graph data for training and testing.
        """

        cycle_graph_data = {'train_cells': {}, 'test_cells': {}}

        for cell_type, cells in preprocessed_data.items():
            for cell_name, cell_data in cells.items():
                print(f"Preparing graph data for {cell_type}/{cell_name}")

                previous_cycle_capacity = self.config['nominal_capacity']

                cycle_graph_data[cell_type][cell_name] = {}

                for cycle_no, cycle_data in cell_data.items():
                    graph = self.construct_graph_for_cycle(cycle_data, previous_cycle_capacity)
                    cycle_graph_data[cell_type][cell_name][cycle_no] = graph

                    # Update previous_cycle_capacity for the next iteration
                    previous_cycle_capacity = cycle_data['ground_truth_cycle']['capacity']

        return cycle_graph_data

    def construct_graph_for_cycle(self, data_point: Dict[str, Any], previous_cycle_capacity: float) -> Data:
        """
        Construct a graph from the provided data point.

        :param data_point: A dictionary containing accumulated time bins, transitions, and ground truth cycle data.
        :param previous_cycle_capacity: The capacity from the previous cycle used as a global feature.
        :return: A PyTorch Geometric Data object representing the graph.
        """

        # Nodes and their features (accumulated time in bins)
        node_features = []
        node_indices = []

        for node, time_spent in data_point['accumulated_time_bins'].items():
            node_features.append([time_spent])
            node_indices.append(node)

        node_features = torch.tensor(node_features, dtype=torch.float)

        # Edges and their features (number of transitions)
        edge_indices = []
        edge_features = []

        for (node_from, node_to), num_jumps in data_point['accumulated_transitions'].items():
            edge_indices.append([node_indices.index(node_from), node_indices.index(node_to)])
            edge_features.append([num_jumps])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)

        # Global feature: previous cycle's capacity
        global_feat = torch.tensor([previous_cycle_capacity], dtype=torch.float)

        # Target: capacity decay
        current_cycle_capacity = data_point['ground_truth_cycle']['capacity']
        capacity_decay = current_cycle_capacity - previous_cycle_capacity
        y = torch.tensor([capacity_decay], dtype=torch.float)

        # Create and return the Data object
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y, global_feat=global_feat)

    def get_bin_mapping(self):
        """
        Create a mapping from flat indices to bin indices for voltage, current, and temperature.

        :return: A dictionary where the keys are flat indices and the values are tuples of bin indices.
        """
        bin_mapping = {}
        for i in range(len(self.voltage_bins)):
            for j in range(len(self.current_bins)):
                for k in range(len(self.temperature_bins)):
                    flat_idx = i * self.num_bins_per_dim[1] * self.num_bins_per_dim[2] + j * self.num_bins_per_dim[2] + k
                    bin_mapping[flat_idx] = (i, j, k)
        return bin_mapping

    def get_bin_intervals(self, v_bin_idx, c_bin_idx, t_bin_idx):
        """
        Get the voltage, current, and temperature intervals for specific bin indices.

        :param v_bin_idx: Index for the voltage bin.
        :param c_bin_idx: Index for the current bin.
        :param t_bin_idx: Index for the temperature bin.
        :return: A tuple containing the voltage, current, and temperature intervals.
        """
        voltage_interval = self.voltage_bins[v_bin_idx]
        current_interval = self.current_bins[c_bin_idx]
        temperature_interval = self.temperature_bins[t_bin_idx]
        return voltage_interval, current_interval, temperature_interval
