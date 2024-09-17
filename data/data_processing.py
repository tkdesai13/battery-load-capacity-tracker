# Description: This file contains the DataProcessor class, which is responsible for loading, preprocessing, and
# structuring the data for training and testing the model.

import os
import itertools
import pickle
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
                raw_cell_data = self.load_mat_file_hdf5storage_(file_path)
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
        min_change = self.config['data']['ground_truth_cycle_validity']['min_capacity_change']
        max_change = self.config['data']['ground_truth_cycle_validity']['max_capacity_change']
        return min_change <= capacity_diff <= max_change

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

    def prepare_scalar_features(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Tuple[
        Dict, Dict[str, Dict[str, float]]]:
        """
        Prepare scalar features from preprocessed data, including min/max calculations for each feature.

        :param preprocessed_data: Preprocessed data organized by cell types and cell names.
        :return: A tuple of all cell target feature data and min/max values for the features.
        """

        # Initialize result containers
        all_cell_target_feature_data = {}
        min_max_values = self._initialize_min_max_trackers()

        time_bins_min_zero_set = False
        transitions_min_zero_set = False

        for cell_type, cells in preprocessed_data.items():
            all_cell_target_feature_data[cell_type] = {}

            for cell_name, cell_data in cells.items():
                print(f"Preparing scalar features for {cell_type}/{cell_name}")
                all_cell_target_feature_data[cell_type][cell_name] = {}
                previous_capacity = self.config['nominal_capacity']  # Starting with nominal capacity

                for cycle_no, cycle_data in sorted(cell_data.items()):
                    cycle_features = self._process_cycle_data(
                        cycle_data, previous_capacity, min_max_values, time_bins_min_zero_set, transitions_min_zero_set
                    )

                    # Store processed data
                    all_cell_target_feature_data[cell_type][cell_name][cycle_no] = cycle_features
                    previous_capacity = cycle_features['capacity']  # Update for the next cycle

        # Ensure this return happens after processing all cell types
        return all_cell_target_feature_data, min_max_values

    def _initialize_min_max_trackers(self) -> Dict[str, Dict[str, float]]:
        """Helper function to initialize min/max trackers for each feature."""
        return {
            'accumulated_time_bins': {'min': float('inf'), 'max': float('-inf')},
            'accumulated_transitions': {'min': float('inf'), 'max': float('-inf')},
            'capacity': {'min': float('inf'), 'max': float('-inf')},
            'capacity_diff': {'min': float('inf'), 'max': float('-inf')},
            'previous_capacity': {'min': float('inf'), 'max': float('-inf')}
        }

    def _process_cycle_data(self, cycle_data: Dict[str, Any], previous_capacity: float,
                            min_max_values: Dict[str, Dict[str, float]],
                            time_bins_min_zero_set: bool, transitions_min_zero_set: bool) -> Dict[str, Any]:
        """
        Process individual cycle data to calculate accumulated features and update min/max values.

        :param cycle_data: Dictionary containing data for a particular cycle.
        :param previous_capacity: Previous cycle's capacity value.
        :param min_max_values: Dictionary to track min/max values for features.
        :return: Processed cycle features.
        """
        # Extract accumulated time bins and transitions
        time_bins = cycle_data.get('accumulated_time_bins', [])
        transitions = cycle_data.get('accumulated_transitions', [])
        current_capacity = cycle_data['ground_truth_cycle'].get('capacity', previous_capacity)

        # Initialize per-cycle data
        cycle_accumulated_time_bins = dict.fromkeys(self.bin_mapping.keys(), 0.0)
        cycle_transition_counts = self.initialize_transitions_per_bin_pair()

        # Process time bins and update accumulated time per bin
        for entry in time_bins:
            bin_index = entry['bin_index']
            time_spent = entry['time_spent']
            cycle_accumulated_time_bins[bin_index] += time_spent

        # Update min/max values for accumulated time bins
        time_bins_min_zero_set = self._update_min_max_for_bins(time_bins, cycle_accumulated_time_bins,
                                                               min_max_values, time_bins_min_zero_set)

        # Process transitions and update counts per bin pair
        for transition in transitions:
            start_bin, end_bin = transition['from_bin'], transition['to_bin']
            cycle_transition_counts[(start_bin, end_bin)] += 1

        # Update min/max values for transitions
        transitions_min_zero_set = self._update_min_max_for_transitions(transitions, cycle_transition_counts,
                                                                        min_max_values, transitions_min_zero_set)

        # Calculate capacity difference
        capacity_diff = current_capacity - previous_capacity

        # Update min/max values for capacity and capacity differences
        self._update_min_max_for_capacity(current_capacity, capacity_diff, previous_capacity, min_max_values)

        # Return cycle-specific feature data
        return {
            'accumulated_time_bins': cycle_accumulated_time_bins,
            'accumulated_transitions': cycle_transition_counts,
            'capacity': current_capacity,
            'date': cycle_data['ground_truth_cycle']['date'],
            'capacity_diff': capacity_diff,
            'previous_capacity': previous_capacity
        }

    def _update_min_max_for_bins(self, time_bins: list, cycle_accumulated_time_bins: dict,
                                 min_max_values: dict, time_bins_min_zero_set: bool) -> bool:
        """
        Update min/max values for accumulated time bins.
        """

        # Update max value for accumulated time bins
        min_max_values['accumulated_time_bins']['max'] = max(
            min_max_values['accumulated_time_bins']['max'], max(cycle_accumulated_time_bins.values())
        )

        # check if min value is zero and set it to zero if not already set to zero else update min value
        if not time_bins_min_zero_set:
            time_bin_indices = {entry['bin_index'] for entry in time_bins}

            # Check if all bins are present in the cycle data and set min to zero
            if len(time_bin_indices) != len(cycle_accumulated_time_bins):
                min_max_values['accumulated_time_bins']['min'] = 0.0
                return True
            else:
                min_max_values['accumulated_time_bins']['min'] = min(
                    min_max_values['accumulated_time_bins']['min'], min(cycle_accumulated_time_bins.values())
                )

        return time_bins_min_zero_set

    def _update_min_max_for_transitions(self, transitions: list, cycle_transition_counts: dict,
                                        min_max_values: dict, transitions_min_zero_set: bool) -> bool:
        """
        Update min/max values for transitions.
        """
        # Update max value for accumulated transitions
        min_max_values['accumulated_transitions']['max'] = max(
            min_max_values['accumulated_transitions']['max'], max(cycle_transition_counts.values())
        )

        # check if min value is zero and set it to zero if not already set to zero else update min value
        if not transitions_min_zero_set:
            transition_pairs = {(transition['from_bin'], transition['to_bin']) for transition in transitions}
            if len(transition_pairs) != len(cycle_transition_counts):
                min_max_values['accumulated_transitions']['min'] = 0
                return True
            else:
                min_max_values['accumulated_transitions']['min'] = min(
                    min_max_values['accumulated_transitions']['min'], min(cycle_transition_counts.values())
                )

        return transitions_min_zero_set

    def _update_min_max_for_capacity(self, current_capacity: float, capacity_diff: float,
                                     previous_capacity: float, min_max_values: dict):
        """
        Update min/max values for capacity and capacity difference.
        """
        min_max_values['capacity']['min'] = min(min_max_values['capacity']['min'], current_capacity)
        min_max_values['capacity']['max'] = max(min_max_values['capacity']['max'], current_capacity)
        min_max_values['capacity_diff']['min'] = min(min_max_values['capacity_diff']['min'], capacity_diff)
        min_max_values['capacity_diff']['max'] = max(min_max_values['capacity_diff']['max'], capacity_diff)
        min_max_values['previous_capacity']['min'] = min(min_max_values['previous_capacity']['min'], previous_capacity)
        min_max_values['previous_capacity']['max'] = max(min_max_values['previous_capacity']['max'], previous_capacity)

    def prepare_vector_features(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict:
        """
        Prepare the data for training and testing by converting the scalar features to vectorized features.

        :param preprocessed_data: The preprocessed data structured as a dictionary.
        :return: The preprocessed data with vector features.
        """

        # Initialize min/max trackers for all metrics
        min_max_values = {
            'accumulated_time_bins': {'min': float('inf'), 'max': float('-inf')},
            'accumulated_transitions': {'min': float('inf'), 'max': float('-inf')},
            'capacity': {'min': float('inf'), 'max': float('-inf')},
            'capacity_diff': {'min': float('inf'), 'max': float('-inf')},
            'previous_capacity': {'min': float('inf'), 'max': float('-inf')},
            'max_transitions': 0
        }

        # Flags to stop min checks after setting to zero
        time_bins_min_zero_set = False
        transitions_min_zero_set = False

        # Initialising the final feature-label dictionary
        all_cell_target_feature_data = {}

        for cell_type, cells in preprocessed_data.items():

            all_cell_target_feature_data[cell_type] = {}

            for cell_name, cell_data in cells.items():
                print(f"Preparing vector features for {cell_type}/{cell_name}")

                all_cell_target_feature_data[cell_type][cell_name] = {}

                previous_capacity = self.config['nominal_capacity']  # Start with nominal capacity

                for cycle_no, cycle_data in cell_data.items():
                    time_bins = cycle_data.get('accumulated_time_bins', {})
                    transition_matrix = cycle_data.get('accumulated_transitions', {})

                    # Update the max sequence length
                    min_max_values['max_transitions'] = max(min_max_values['max_transitions'], len(time_bins))

                    # Convert the scalar time bins to vectorized features
                    vector_time_bins = self.generate_transition_matrix(time_bins)

                    current_capacity = cycle_data['ground_truth_cycle'].get('capacity', previous_capacity)
                    # Calculate capacity difference
                    capacity_diff = current_capacity - previous_capacity

                    # Update the cycle data with vector features
                    cell_data[cycle_no]['accumulated_time_bins'] = vector_time_bins
                    cell_data[cycle_no]['accumulated_transitions'] = transition_matrix

                    # Update min/max for capacity, capacity_diff, and previous_capacity
                    min_max_values['capacity']['min'] = min(min_max_values['capacity']['min'], current_capacity)
                    min_max_values['capacity']['max'] = max(min_max_values['capacity']['max'], current_capacity)
                    min_max_values['capacity_diff']['min'] = min(min_max_values['capacity_diff']['min'], capacity_diff)
                    min_max_values['capacity_diff']['max'] = max(min_max_values['capacity_diff']['max'], capacity_diff)
                    min_max_values['previous_capacity']['min'] = min(min_max_values['previous_capacity']['min'],
                                                                     previous_capacity)
                    min_max_values['previous_capacity']['max'] = max(min_max_values['previous_capacity']['max'],
                                                                     previous_capacity)

                    # Feature-label dictionary
                    all_cell_target_feature_data[cell_type][cell_name][cycle_no] = {
                        'accumulated_time_bins': vector_time_bins,
                        'accumulated_transitions': transition_matrix,
                        'capacity': current_capacity,
                        'date': cycle_data['ground_truth_cycle']['date'],
                        'capacity_diff': capacity_diff,
                        'previous_capacity': previous_capacity
                    }

                    previous_capacity = current_capacity

        return all_cell_target_feature_data, min_max_values

    def generate_transition_matrix(self, time_bins: dict) -> np.ndarray:
        """
        Generate a matrix of shape (no_of_transitions, no_of_bins) where each row corresponds to a transition
        and only the specific bin column has the time spent.

        :param time_bins: A list of dictionaries with keys 'bin_index', 'time_spent', 'transition_index'.
        :return: A numpy matrix of shape (no_of_transitions, no_of_bins).
        """
        # Total number of bins (based on bin_mapping) and transitions
        no_of_bins = len(self.bin_mapping)
        no_of_transitions = len(time_bins)

        # Initialize a zero matrix of shape (no_of_transitions, no_of_bins)
        matrix = np.zeros((no_of_transitions, no_of_bins))

        # Reverse mapping from bin_mapping keys to column index
        bin_mapping_index = {key: idx for idx, key in enumerate(self.bin_mapping.keys())}

        # Iterate over each transition and assign time_spent to the correct bin column
        for i, transition in enumerate(time_bins):
            bin_index = transition['bin_index']  # This is the bin key, a tuple like (0, 0, 0)
            time_spent = transition['time_spent']

            # Find the corresponding column in the matrix using bin_mapping_index
            if bin_index in bin_mapping_index:
                column_idx = bin_mapping_index[bin_index]
                matrix[i, column_idx] = time_spent

        return matrix

    def scale_data(self, all_cell_target_feature_data: Dict[str, Dict[str, Any]], min_max_values: Dict) -> Dict:
        """
        Scale the data using the global min/max values for each feature.

        :param all_cell_target_feature_data: The preprocessed data structured as a dictionary.
        :param min_max_values: The global min/max values for each feature.
        :return: The scaled data or the original data based on the configuration.
        """

        # Check if data_scaling is enabled in the config
        if not self.config['model_config'].get('data_scaling', False):
            # If data_scaling is not enabled, return the data as is
            return all_cell_target_feature_data

        # Initialize scaled data dictionary
        scaled_data = {}

        # Features to be scaled according to config
        signals_to_scale = self.config['model_config'].get('signals_to_scale', [])

        for cell_type, cells in all_cell_target_feature_data.items():
            scaled_data[cell_type] = {}
            for cell_name, cycles in cells.items():
                scaled_data[cell_type][cell_name] = {}

                for cycle_no, cycle_data in cycles.items():
                    scaled_cycle_data = {}

                    # Iterate over all signals in cycle_data
                    for signal in cycle_data.keys():
                        if signal in signals_to_scale:
                            # Apply scaling if the signal is in signals_to_scale
                            min_value = min_max_values[signal]['min']
                            max_value = min_max_values[signal]['max']

                            if isinstance(cycle_data[signal], dict):  # Handle dictionary values (like bins)
                                scaled_signal_data = {}
                                for key, value in cycle_data[signal].items():
                                    scaled_signal_data[key] = self._scale_value(value, min_value, max_value)
                                scaled_cycle_data[signal] = scaled_signal_data
                            else:  # Handle scalar values
                                scaled_cycle_data[signal] = self._scale_value(cycle_data[signal], min_value, max_value)
                        else:
                            # If the signal is not to be scaled, copy it as is
                            scaled_cycle_data[signal] = cycle_data[signal]

                    scaled_data[cell_type][cell_name][cycle_no] = scaled_cycle_data

        return scaled_data

    def _scale_value(self, value, min_value, max_value):
        if max_value > min_value:
            return (value - min_value) / (max_value - min_value)
        else:
            return 0.0  # Avoid division by zero

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
            "train": {
                "timebins": [],
                'previous_capacity': [],
                "capacity_difference": []
            },
            "validation": {
                "timebins": [],
                'previous_capacity': [],
                "capacity_difference": []
            },
            "test_cells": train_test_data["test_cells"]  # Directly assign test data without shuffling
        }

        # Combine all training cycles into a single list of tuples (timebins, previous_capacity, capacity_difference)
        combined_cycles = []
        for cell_name, data in train_test_data["train_cells"].items():
            for cycle_no, cycle_data in data.items():
                # Handle both dict and numpy array formats
                accumulated_time_bins = cycle_data["accumulated_time_bins"]
                if isinstance(accumulated_time_bins, dict):
                    accumulated_time_bins = list(accumulated_time_bins.values())  # Convert dict_values to list
                elif isinstance(accumulated_time_bins, np.ndarray):
                    accumulated_time_bins = accumulated_time_bins.tolist()  # Convert ndarray to list

                combined_cycles.append((
                    accumulated_time_bins,  # Ensure it's a list in both cases
                    cycle_data["previous_capacity"],
                    cycle_data["capacity_diff"]
                ))

        # Shuffle the list of combined cycles
        np.random.shuffle(combined_cycles)

        # Calculate the split index for validation
        val_split_ratio = self.config['model_specifications']['val_train_split_ratio']
        split_index = int(len(combined_cycles) * (1 - val_split_ratio))

        # Split the cycles into training and validation sets
        train_cycles = combined_cycles[:split_index]
        val_cycles = combined_cycles[split_index:]

        # Unzip the training and validation cycles into their respective components
        split_data["train"]["timebins"], split_data["train"]["previous_capacity"], \
            split_data["train"]["capacity_difference"] = zip(*train_cycles) if train_cycles else ([], [], [])
        split_data["validation"]["timebins"], split_data["validation"]["previous_capacity"], \
            split_data["validation"]["capacity_difference"] = zip(*val_cycles) if val_cycles else ([], [], [])

        # Convert the zipped results back to lists
        split_data["train"]["timebins"] = list(split_data["train"]["timebins"])
        split_data["train"]["previous_capacity"] = list(split_data["train"]["previous_capacity"])
        split_data["train"]["capacity_difference"] = list(split_data["train"]["capacity_difference"])

        split_data["validation"]["timebins"] = list(split_data["validation"]["timebins"])
        split_data["validation"]["previous_capacity"] = list(split_data["validation"]["previous_capacity"])
        split_data["validation"]["capacity_difference"] = list(
            split_data["validation"]["capacity_difference"])

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
