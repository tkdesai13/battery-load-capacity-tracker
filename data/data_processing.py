import os
import itertools
import pickle
import random
from datetime import datetime
import scipy.io
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple, List

# import torch
# import torch.nn as nn
# import torch_geometric
# from torch_geometric.data import Data

# Set random seeds for reproducibility
seed = 42
rng = np.random.default_rng(seed=seed)


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
        Iterates through each pickle file in the data_path, processes each batch of cell data,
        and structures the processed data into a dictionary, including batch numbers.

        :return: The preprocessed data structured as a dictionary, with batch numbers as cell types.
        """
        data_path = self.config['paths']['data']
        preprocessed_data = {}

        # Iterate over all .pkl files in the data directory
        all_files = os.listdir(data_path)
        for file_name in all_files:
            if file_name.endswith('.pkl'):
                batch_no = self._extract_batch_number(file_name)  # Function to extract batch number from the file name
                print(f"Processing batch {batch_no} from {file_name}")

                # Load the pickle file
                file_path = os.path.join(data_path, file_name)
                with open(file_path, 'rb') as f:
                    batch_data = pickle.load(f)

                # Initialize batch in preprocessed_data
                preprocessed_data[batch_no] = {}

                # Process each cell in the batch
                for cell_name, cell_data in batch_data.items():
                    print(f"Processing cell {cell_name} in batch {batch_no}")

                    # Send the cell data to the preprocess function
                    processed_cell_data = self._process_cell_data(cell_data, batch_no)

                    # Store the processed cell data under the batch_no
                    preprocessed_data[batch_no][cell_name] = processed_cell_data

        return preprocessed_data

    def _extract_batch_number(self, file_name: str) -> str:
        """
        Helper function to extract the batch number from the file name.
        Assumes that the batch number is in the format 'Sbatch_bX.pkl', where X is the batch number.

        :param file_name: The file name from which to extract the batch number.
        :return: The extracted batch number as a string.
        """
        # Assuming the file name is in the format 'Sbatch_bX.pkl'
        batch_no = file_name.split('_')[1].replace('.pkl', '')  # Extract 'bX'
        return batch_no

    def _get_field_mapping(self) -> Dict[str, str]:
        """
        Generate a mapping from raw data keys to desired field names using self.config.
        """
        field_mapping = {}
        for raw_key, signal_info in self.config['signals'].items():
            field_mapping[raw_key] = signal_info['name']
        return field_mapping

    def _initialize_accumulated_data(self) -> dict:
        """
        Initialize the structure for accumulating data using field names from self.config.
        """
        field_mapping = self._get_field_mapping()
        accumulated_data = {field_name: [] for field_name in field_mapping.values()}
        return accumulated_data

    def _accumulate_cycles_data(self, accumulated_data: dict, cycle_data: dict):
        """
        Accumulate data from a single step into the accumulated data structure using field mapping.
        """
        field_mapping = self._get_field_mapping()
        for raw_key, field_name in field_mapping.items():
            if raw_key in cycle_data:
                value = cycle_data[raw_key]
                if isinstance(value, list):
                    accumulated_data[field_name].extend(value)
                else:
                    accumulated_data[field_name].append(value)

    def _process_cell_data(self, cell_data: Dict[str, Any], batch_no: str) -> Dict[int, Dict[str, Any]]:
        """
        Preprocess the data for a single cell.
        """
        cumulative_time_for_cell = 0.0
        processed_data = {}
        accumulated_data = self._initialize_accumulated_data()

        for cycle_counter in cell_data['cycles'].keys():

            cycle_data = cell_data['cycles'][cycle_counter]

            # Access the dictionary of batches and cycles to skip
            batches_to_skip = self.config['data']['batches_and_cycle_to_skip']

            # Check if the current batch has a cycle to skip and if the current cycle matches
            if batch_no in batches_to_skip and int(cycle_counter) == batches_to_skip[batch_no]:
                continue  # Skip processing this cycle

            self._accumulate_cycles_data(accumulated_data, cycle_data)

            ground_truth_cycle_frequency = self.config['data']['ground_truth_cycle_frequency']
            cycle_counter_from_one = int(cycle_counter) + 1
            if cycle_counter_from_one % ground_truth_cycle_frequency == 0:
                # if cycle_counter > 1 and not self._is_valid_capacity_change(processed_data, cycle_counter, cycle_data):
                #     continue

                processed_load_pattern, cumulative_time_for_cell = self._process_load_pattern(accumulated_data,
                                                                                              cumulative_time_for_cell)

                cell_summery_data = cell_data['summary']

                processed_data[cycle_counter] = {
                    'accumulated_time_bins': processed_load_pattern["time_bins"],
                    'accumulated_transitions': processed_load_pattern["transitions"],
                    'ground_truth_cycle': self._process_reference_capacity(cell_summery_data=cell_summery_data,
                                                                           cycle_counter=cycle_counter),
                    'cumulative_time_at_ground_truth': cumulative_time_for_cell
                }

                accumulated_data = self._initialize_accumulated_data()

        return processed_data

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

    def _process_load_pattern(self, step_data: Dict[str, Any], cumulative_time_for_cell: float) -> tuple[
        dict[str, list], float]:
        """
        Process the load pattern data by calculating the time spent in each bin and transitions between bins.
        This includes adjusting the time data to form a continuous time series across cycles.
        """

        # Adjust and concatenate time data
        adjusted_time_arrays = []
        for time_array in step_data["time"]:
            time_array = time_array / 60  # Convert time to hours
            adjusted_time_array = time_array + cumulative_time_for_cell
            adjusted_time_arrays.append(adjusted_time_array)  # Adjust time to be continuous across cycles
            cumulative_time_for_cell += time_array[0, -1]  # Update cumulative time for the cell

        time_data = np.concatenate(adjusted_time_arrays, axis=1).flatten()

        # Concatenate other data arrays
        voltage_data = np.concatenate(step_data["voltage"], axis=1).flatten()
        current_data = np.concatenate(step_data["current"], axis=1).flatten()
        temperature_data = np.concatenate(step_data["temperature"], axis=1).flatten()

        # Pass the data to the _bin_data_vector function
        time_bins, transitions = self._bin_data_vector(voltage_data, current_data, temperature_data, time_data)
        return {"time_bins": time_bins, "transitions": transitions}, cumulative_time_for_cell

    def _bin_data_vector(self, voltage, current, temperature, time) -> Tuple[list, list]:
        """
        Bin the battery data and calculate the vectorized time spent in each bin and transitions between bins.
        """
        v_indices = self._digitize_data(voltage, self.voltage_bins)
        c_indices = self._digitize_data(current, self.current_bins)
        t_indices = self._digitize_data(temperature, self.temperature_bins)

        combined_indices = np.stack((v_indices, c_indices, t_indices), axis=-1)
        time_diffs = np.diff(time) / 60  # Convert time differences to hours

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

    def _process_reference_capacity(self, cell_summery_data: Dict[str, Any], cycle_counter: int) -> Dict[str, Any]:
        """
        Process the capacity measurement data.
        """
        # date_str = cell_summery_data.get("date", "")
        # step_date = datetime.strptime(date_str, "%d-%b-%Y %H:%M:%S") if date_str else None

        # current_data = cell_summery_data["current"]
        # relative_time_data = cell_summery_data["relativeTime"]
        # relative_time_hours = np.array(relative_time_data) / 3600.0

        # capacity = np.trapz(current_data, relative_time_hours)
        cycle_counter_from_one = int(cycle_counter) + 1
        capacity = cell_summery_data["QD"][int(cycle_counter)]

        return {"cycle_no": cycle_counter_from_one, "capacity": capacity}

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

                for cycle_no, cycle_data in cell_data.items():
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
            'bin_indices': list(self.bin_mapping.keys()),
            'accumulated_time_bins': cycle_accumulated_time_bins,
            'accumulated_transitions': cycle_transition_counts,
            'capacity': current_capacity,
            'cumulative_time_at_ground_truth': cycle_data['cumulative_time_at_ground_truth'],
            'cycle_no': cycle_data['ground_truth_cycle']['cycle_no'],
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
                    bin_indices, vector_time_bins = self.generate_transition_matrix(time_bins)

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
                        'bin_indices': bin_indices,
                        'accumulated_time_bins': vector_time_bins,
                        'accumulated_transitions': transition_matrix,
                        'capacity': current_capacity,
                        'cumulative_time_at_ground_truth': cycle_data['cumulative_time_at_ground_truth'],
                        'cycle_no': cycle_data['ground_truth_cycle']['cycle_no'],
                        'capacity_diff': capacity_diff,
                        'previous_capacity': previous_capacity
                    }

                    previous_capacity = current_capacity

        return all_cell_target_feature_data, min_max_values

    def prepare_vector_features_embedded(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Tuple[Dict, Dict]:
        """
        Prepare the data for training and testing by converting the scalar features to vectorized features.

        :param preprocessed_data: The preprocessed data structured as a dictionary.
        :return: The preprocessed data with vector features and min/max values.
        """

        # Initialize min/max trackers for all metrics
        min_max_values = {
            'capacity': {'min': float('inf'), 'max': float('-inf')},
            'capacity_diff': {'min': float('inf'), 'max': float('-inf')},
            'previous_capacity': {'min': float('inf'), 'max': float('-inf')},
            'max_sequence_length': 0
        }

        # Initialize the final feature-label dictionary
        all_cell_target_feature_data = {}

        for cell_type, cells in preprocessed_data.items():

            all_cell_target_feature_data[cell_type] = {}

            for cell_name, cell_data in cells.items():
                print(f"Preparing vector features for {cell_type}/{cell_name}")

                all_cell_target_feature_data[cell_type][cell_name] = {}

                previous_capacity = self.config['nominal_capacity']  # Start with nominal capacity

                for cycle_no, cycle_data in cell_data.items():
                    time_bins = cycle_data.get('accumulated_time_bins', [])
                    transition_matrix = cycle_data.get('accumulated_transitions', {})

                    # Generate bin indices and time spent sequences
                    bin_indices, time_spent_values = self.generate_bin_index_and_time_sequences(time_bins)
                    cell_data[cycle_no]['accumulated_transitions'] = transition_matrix

                    # Update the max sequence length
                    sequence_length = len(bin_indices)
                    min_max_values['max_sequence_length'] = max(min_max_values['max_sequence_length'], sequence_length)

                    current_capacity = cycle_data['ground_truth_cycle'].get('capacity', previous_capacity)
                    # Calculate capacity difference
                    capacity_diff = current_capacity - previous_capacity

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
                        'bin_indices': bin_indices,
                        'accumulated_time_bins': time_spent_values,
                        'accumulated_transitions': transition_matrix,
                        'capacity': current_capacity,
                        'cumulative_time_at_ground_truth': cycle_data['cumulative_time_at_ground_truth'],
                        'cycle_no': cycle_data['ground_truth_cycle']['cycle_no'],
                        'capacity_diff': capacity_diff,
                        'previous_capacity': previous_capacity
                    }

                    previous_capacity = current_capacity

        return all_cell_target_feature_data, min_max_values

    def generate_bin_index_and_time_sequences(self, time_bins: List[Dict]) -> Tuple[List[int], List[float]]:
        """
        Generate sequences of bin indices and time spent values from time_bins data.

        :param time_bins: A list of dictionaries with keys 'bin_index', 'time_spent', 'transition_index'.
        :return: Two lists: bin_indices (List[int]) and time_spent (List[float]).
        """
        bin_indices = []
        time_spent_values = []

        # Reverse mapping from bin_mapping keys to bin indices
        bin_mapping_index = {key: idx for idx, key in enumerate(self.bin_mapping.keys(), start=1)}  # Start from 1

        for transition in time_bins:
            bin_key = transition['bin_index']  # This is the bin key, a tuple like (0, 0, 0)
            time_spent = transition['time_spent']

            # Get the bin index (integer) corresponding to the bin key
            bin_index = bin_mapping_index.get(bin_key, 0)  # Use 0 if bin_key is not found

            bin_indices.append(bin_index)
            time_spent_values.append(time_spent)

        return bin_indices, time_spent_values

    def generate_transition_matrix(self, time_bins: dict) -> Tuple[List[int], np.ndarray]:
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

        bin_indices = []
        for i, transition in enumerate(time_bins):
            bin_index = transition['bin_index']  # This is the bin key, a tuple like (0, 0, 0)
            time_spent = transition['time_spent']

            # Find the corresponding column in the matrix using bin_mapping_index
            if bin_index in bin_mapping_index:
                column_idx = bin_mapping_index[bin_index]
                matrix[i, column_idx] = time_spent
                bin_indices.append(column_idx)

        return bin_indices, matrix

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

    def shuffle_and_split_data(self, scaled_data: Dict[str, Dict[str, Dict[str, Any]]]) -> (
            Tuple)[Dict[str, Dict[str, Any]], List[str]]:
        """
        Split data into train, validation, and test sets based on cell and cycle data.

        1. Randomly select % (according to config file) of the cells for the test set (without shuffling their cycles).
        2. Combine cycles from the remaining of cells, shuffle them, and split 80% ((according to config file)) for training and 20% for validation.

        :param scaled_data: The dataset structured by batch, cell, and cycles.
        :return: A dictionary with the split data for train, validation, and test sets.
        """

        split_data = {
            "train": {
                "bin_indices": [],
                "timebins": [],
                "previous_capacity": [],
                "capacity_difference": [],
                "cumulative_time_at_ground_truth": []
            },
            "validation": {
                "bin_indices": [],
                "timebins": [],
                "previous_capacity": [],
                "capacity_difference": [],
                "cumulative_time_at_ground_truth": []
            },
            "test": {}  # This will store test cells
        }

        for batch_name, batch_cells in scaled_data.items():
            cell_names = list(batch_cells.keys())

            # Split test cells and train/validation cells
            test_cells, train_cells = self._split_test_cells(cell_names)

            # Assign test cells directly to the test set
            for test_cell in test_cells:
                split_data["test"][test_cell] = batch_cells[test_cell]

            # Process train and validation cycles
            combined_cycles = self._combine_cycles_from_cells(train_cells, batch_cells)

            # Shuffle and split combined cycles into training and validation sets
            train_cycles, val_cycles = self._split_train_val_cycles(combined_cycles)

            # Add the cycles to the respective train and validation sets
            self._add_cycles_to_split(split_data, "train", train_cycles)
            self._add_cycles_to_split(split_data, "validation", val_cycles)

        return split_data, list(split_data['test'].keys())

    def prepare_data_sequence_to_sequence(self, scaled_data: Dict[str, Dict[str, Dict[str, Any]]]) -> Tuple[
        Dict[str, Dict[str, Any]], List[str]]:
        """
        Prepare data for sequence-to-sequence learning, maintaining temporal order of cycles.

        1. Split cells into training, validation, and test sets.
        2. For each cell, divide cycles into encoder (input) and decoder (output) sequences.
        3. Return split data with train, validation, and test sets.

        :param scaled_data: The dataset structured by batch, cell, and cycles.
        :return: A dictionary with train, validation, and test data splits and the list of test cell names.
        """

        # Initialize the data split
        split_data = {
            "train": {},
            "validation": {},
            "test": {}
        }

        # Configuration parameters
        test_percentage = self.config['model_specifications']['test_train_cells_ratio']  # Fraction of cells for testing
        val_percentage = self.config['model_specifications'][
            'val_train_cycles_ratio']  # Fraction of training for validation
        encoder_length = self.config['model_specifications']['encoder_input_seq']  # Fixed number of cycles for encoder

        for batch_name, batch_cells in scaled_data.items():
            cell_names = list(batch_cells.keys())

            # Use RNG to shuffle deterministically
            shuffled_indices = rng.permutation(len(cell_names))
            cell_names = [cell_names[i] for i in shuffled_indices]

            # Split cells into test, train, and validation
            num_test_cells = int(len(cell_names) * test_percentage)
            test_cells = cell_names[:num_test_cells]
            train_val_cells = cell_names[num_test_cells:]

            num_val_cells = int(len(train_val_cells) * val_percentage)
            val_cells = train_val_cells[:num_val_cells]
            train_cells = train_val_cells[num_val_cells:]

            # Process test cells
            for cell_name in test_cells:
                self._process_cell_cycles(
                    cell_name,
                    batch_cells[cell_name],
                    encoder_length,
                    split_data,
                    "test"
                )

            # Process train and validation cells
            for cell, split_key in zip([train_cells, val_cells], ["train", "validation"]):
                for cell_name in cell:
                    self._process_cell_cycles(
                        cell_name,
                        batch_cells[cell_name],
                        encoder_length,
                        split_data,
                        split_key
                    )

        return split_data, test_cells

    def _process_cell_cycles(self, cell_name, cell_data, encoder_length, split_data, split_key):
        """
        Process a single cell's cycles to extract encoder-decoder sequences with predefined split ratios.

        :param cell_name: The name of the cell being processed.
        :param cell_data: The data for a single cell.
        :param encoder_length: Minimum number of cycles for the encoder (not used directly now as we rely on ratios).
        :param split_data: Dictionary to store processed data.
        :param split_key: Key indicating the data split (train, validation, or test).
        """
        cycles = list(cell_data.keys())
        cycles.sort(key=int)  # Ensure sequential order

        max_no_of_sequences_per_cell = self.config['model_specifications']['max_no_of_sequences_per_cell']
        min_decoder_length = self.config['model_specifications'].get('min_decoder_length', 1)

        total_cycles = len(cycles)
        # if total_cycles <= encoder_length + min_decoder_length:
        #     return  # Skip cells with insufficient cycles

        # Initialize structure for this cell in the split if not present
        if cell_name not in split_data[split_key]:
            split_data[split_key][cell_name] = {
                "encoder_inputs": {
                    "bin_indices": [],
                    "timebins": [],
                    "capacity": [],
                    "previous_capacity": [],
                    "capacity_difference": [],
                    "cumulative_time": [],
                    "cycle_no": []
                },
                "decoder_inputs": {
                    "bin_indices": [],
                    "timebins": [],
                    "capacity": [],
                    "previous_capacity": [],
                    "cumulative_time": [],
                    "cycle_no": [],
                },
                "decoder_outputs": {
                    "capacity_difference": []
                },
            }

        # Retrieve ratio map from config
        ratio_map = self.config.get('encoder_decoder_ratio_map', {})
        # Convert max_no_of_sequences_per_cell to string since keys in config are strings
        max_no_str = str(max_no_of_sequences_per_cell)

        if max_no_str in ratio_map:
            split_ratios = ratio_map[max_no_str]
        else:
            # If not found, default to a single split at 50%
            split_ratios = [0.5]

        # Convert ratios to actual split points
        sequence_split_points = []
        for ratio in split_ratios:
            split_point = int(round(total_cycles * ratio))
            # Ensure minimum decoder length
            if split_point < total_cycles - min_decoder_length:
                sequence_split_points.append(split_point)

        for split_point in sequence_split_points:
            encoder_cycles = cycles[:split_point]  # Encoder: cycles from start to split_point
            decoder_cycles = cycles[split_point:]  # Decoder: cycles from split_point onward

            # Ensure decoder length is sufficient
            if len(decoder_cycles) < min_decoder_length:
                continue

            # Encoder inputs
            encoder_bin_indices = [cell_data[c]['bin_indices'] for c in encoder_cycles]
            encoder_timebins = [cell_data[c]['accumulated_time_bins'] for c in encoder_cycles]
            encoder_capacity_difference = [cell_data[c]['capacity_diff'] for c in encoder_cycles]
            encoder_capacity = [cell_data[c]['capacity'] for c in encoder_cycles]
            encoder_previous_capacity = [cell_data[c]['previous_capacity'] for c in encoder_cycles]
            encoder_cumulative_time = [cell_data[c]['cumulative_time_at_ground_truth'] for c in encoder_cycles]
            encoder_cycle_no = [cell_data[c]['cycle_no'] for c in encoder_cycles]

            # Decoder inputs and outputs
            decoder_bin_indices = [cell_data[c]['bin_indices'] for c in decoder_cycles]
            decoder_timebins = [cell_data[c]['accumulated_time_bins'] for c in decoder_cycles]
            decoder_capacity_difference = [cell_data[c]['capacity_diff'] for c in decoder_cycles]
            decoder_capacity = [cell_data[c]['capacity'] for c in decoder_cycles]
            decoder_previous_capacity = [cell_data[c]['previous_capacity'] for c in decoder_cycles]
            decoder_cumulative_time = [cell_data[c]['cumulative_time_at_ground_truth'] for c in decoder_cycles]
            decoder_cycle_no = [cell_data[c]['cycle_no'] for c in decoder_cycles]

            # Append to the respective cell's structure
            split_data[split_key][cell_name]["encoder_inputs"]["bin_indices"].append(encoder_bin_indices)
            split_data[split_key][cell_name]["encoder_inputs"]["timebins"].append(encoder_timebins)
            split_data[split_key][cell_name]["encoder_inputs"]["capacity_difference"].append(
                encoder_capacity_difference)
            split_data[split_key][cell_name]["encoder_inputs"]["capacity"].append(encoder_capacity)
            split_data[split_key][cell_name]["encoder_inputs"]["previous_capacity"].append(encoder_previous_capacity)
            split_data[split_key][cell_name]["encoder_inputs"]["cumulative_time"].append(encoder_cumulative_time)
            split_data[split_key][cell_name]["encoder_inputs"]["cycle_no"].append(encoder_cycle_no)

            split_data[split_key][cell_name]["decoder_inputs"]["bin_indices"].append(decoder_bin_indices)
            split_data[split_key][cell_name]["decoder_inputs"]["timebins"].append(decoder_timebins)
            split_data[split_key][cell_name]["decoder_outputs"]["capacity_difference"].append(
                decoder_capacity_difference)
            split_data[split_key][cell_name]["decoder_inputs"]["capacity"].append(decoder_capacity)
            split_data[split_key][cell_name]["decoder_inputs"]["previous_capacity"].append(decoder_previous_capacity)
            split_data[split_key][cell_name]["decoder_inputs"]["cumulative_time"].append(decoder_cumulative_time)
            split_data[split_key][cell_name]["decoder_inputs"]["cycle_no"].append(decoder_cycle_no)

    def shuffle_and_split_data_sequence_to_sequence(self, train_val_test_data):
        """
        Combine and shuffle sequences across cells for train, validation, and test splits.
        Calculate max and min sequence lengths for encoder and decoder.

        :param train_val_test_data: The split data structured by train, validation, and test sets.
        :return: Shuffled data and statistics on sequence lengths.
        """
        # Dynamically extract keys from the first available cell
        example_cell = next(iter(train_val_test_data["train"].values()))
        encoder_input_keys = example_cell["encoder_inputs"].keys()
        decoder_input_keys = example_cell["decoder_inputs"].keys()
        decoder_output_keys = example_cell["decoder_outputs"].keys()

        # Initialize the shuffled data structure
        shuffled_data = self._initialize_shuffled_structure(encoder_input_keys, decoder_input_keys,
                                                            decoder_output_keys)

        # Initialize variables to track max and min sequence lengths
        seq_lengths = {split: {"encoder_max": 0, "encoder_min": float("inf"),
                               "decoder_max": 0, "decoder_min": float("inf")} for split in
                       ["train", "validation", "test"]}

        # Process each split
        for split_key in ["train", "validation", "test"]:
            for cell_name, cell_data in train_val_test_data[split_key].items():
                num_sequences = int(self.config["model_specifications"]["max_no_of_sequences_per_cell"])
                for i in range(num_sequences):
                    # Add sequences to shuffled_data
                    self._append_sequences(shuffled_data[split_key], cell_data, encoder_input_keys,
                                           decoder_input_keys, decoder_output_keys, i)

                    # Update sequence length stats
                    encoder_seq = cell_data["encoder_inputs"]["bin_indices"][i]
                    decoder_seq = cell_data["decoder_inputs"]["bin_indices"][i]
                    self._update_seq_lengths(seq_lengths[split_key], encoder_seq, decoder_seq)

            # Shuffle the sequences within the split
            self._shuffle_split(shuffled_data[split_key], encoder_input_keys, decoder_input_keys,
                                decoder_output_keys)

        return shuffled_data, seq_lengths

    def _initialize_shuffled_structure(self, encoder_input_keys, decoder_input_keys, decoder_output_keys):
        """Initialize the shuffled data structure dynamically based on the keys."""
        return {
            split: {
                "encoder_inputs": {key: [] for key in encoder_input_keys},
                "decoder_inputs": {key: [] for key in decoder_input_keys},
                "decoder_outputs": {key: [] for key in decoder_output_keys},
            } for split in ["train", "validation", "test"]
        }

    def _append_sequences(self, split_data, cell_data, encoder_input_keys, decoder_input_keys, decoder_output_keys,
                          idx):
        """Append sequences from a cell to the split data."""
        for key in encoder_input_keys:
            split_data["encoder_inputs"][key].append(cell_data["encoder_inputs"][key][idx])
        for key in decoder_input_keys:
            split_data["decoder_inputs"][key].append(cell_data["decoder_inputs"][key][idx])
        for key in decoder_output_keys:
            split_data["decoder_outputs"][key].append(cell_data["decoder_outputs"][key][idx])

    def _update_seq_lengths(self, seq_length_stats, encoder_seq, decoder_seq):
        """Update max and min sequence lengths for a split."""
        encoder_len = len(encoder_seq)
        decoder_len = len(decoder_seq)
        seq_length_stats["encoder_max"] = max(seq_length_stats["encoder_max"], encoder_len)
        seq_length_stats["encoder_min"] = min(seq_length_stats["encoder_min"], encoder_len)
        seq_length_stats["decoder_max"] = max(seq_length_stats["decoder_max"], decoder_len)
        seq_length_stats["decoder_min"] = min(seq_length_stats["decoder_min"], decoder_len)

    def _shuffle_split(self, split_data, encoder_input_keys, decoder_input_keys, decoder_output_keys):
        """Shuffle the sequences within a split without mixing keys."""
        # Extract the entire sequence for each sample (grouped by index) to maintain alignment
        num_samples = len(split_data["encoder_inputs"][next(iter(encoder_input_keys))])  # Get the number of samples
        combined_sequences = [
            {
                "encoder_inputs": {key: split_data["encoder_inputs"][key][i] for key in encoder_input_keys},
                "decoder_inputs": {key: split_data["decoder_inputs"][key][i] for key in decoder_input_keys},
                "decoder_outputs": {key: split_data["decoder_outputs"][key][i] for key in decoder_output_keys},
            }
            for i in range(num_samples)
        ]

        # Shuffle the combined sequences using NumPy's default_rng for reproducibility
        shuffled_indices = rng.permutation(len(combined_sequences))
        combined_sequences = [combined_sequences[i] for i in shuffled_indices]

        # Unpack shuffled sequences back into the original structure
        for key in encoder_input_keys:
            split_data["encoder_inputs"][key] = [seq["encoder_inputs"][key] for seq in combined_sequences]
        for key in decoder_input_keys:
            split_data["decoder_inputs"][key] = [seq["decoder_inputs"][key] for seq in combined_sequences]
        for key in decoder_output_keys:
            split_data["decoder_outputs"][key] = [seq["decoder_outputs"][key] for seq in combined_sequences]

    def _split_test_cells(self, cell_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split cell names into test and training/validation sets.

        :param cell_names: List of cell names in a batch.
        :return: Tuple of (test_cells, train_cells).
        """
        test_train_cells_ratio = self.config['model_specifications']['test_train_cells_ratio']
        test_size = int(len(cell_names) * test_train_cells_ratio)

        # Use numpy RNG for deterministic, cross-platform sampling
        test_cells = rng.choice(cell_names, size=test_size, replace=False).tolist()

        train_cells = [cell for cell in cell_names if cell not in test_cells]
        return test_cells, train_cells

    def _combine_cycles_from_cells(self, cell_names: List[str], batch_cells: Dict[str, Any]) -> List[Tuple]:
        """
        Combine cycles from cells into a single list for further processing.

        :param cell_names: List of cell names to combine.
        :param batch_cells: All cell data for the current batch.
        :return: A list of combined cycles from the given cells.
        """
        combined_cycles = []

        for cell in cell_names:
            for cycle_data in batch_cells[cell].values():
                bin_indices = self._convert_to_list(cycle_data["bin_indices"])
                accumulated_time_bins = self._convert_to_list(cycle_data["accumulated_time_bins"])
                previous_capacity = cycle_data["previous_capacity"]
                capacity_diff = cycle_data["capacity_diff"]
                cumulative_time_at_ground_truth = cycle_data["cumulative_time_at_ground_truth"]

                combined_cycles.append((
                    bin_indices,
                    accumulated_time_bins,
                    previous_capacity,
                    capacity_diff,
                    cumulative_time_at_ground_truth
                ))

        return combined_cycles

    def _split_train_val_cycles(self, combined_cycles: List[Tuple]) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Split the combined cycles into training and validation sets.

        :param combined_cycles: List of cycles to split.
        :return: Tuple of (train_cycles, val_cycles).
        """

        # Shuffle cycles deterministically
        shuffled_indices = rng.permutation(len(combined_cycles))
        combined_cycles = [combined_cycles[i] for i in shuffled_indices]

        # Determine split index
        val_split_ratio = self.config['model_specifications']['val_train_cycles_ratio']
        split_index = int(len(combined_cycles) * (1 - val_split_ratio))

        # Split into training and validation sets
        train_cycles = combined_cycles[:split_index]
        val_cycles = combined_cycles[split_index:]

        return train_cycles, val_cycles

    def _add_cycles_to_split(self, split_data: Dict[str, Any], split_type: str, cycles: List[Tuple]) -> None:
        """
        Add cycles to the specified split (train or validation) in split_data.

        :param split_data: Dictionary containing split data.
        :param split_type: 'train' or 'validation' to determine where to add the data.
        :param cycles: List of cycle data tuples to add.
        """
        if cycles:
            bin_indices, timebins, previous_capacity, capacity_difference, cumulative_time_at_ground_truth = zip(
                *cycles)
            split_data[split_type]["bin_indices"].extend(bin_indices)
            split_data[split_type]["timebins"].extend(timebins)
            split_data[split_type]["previous_capacity"].extend(previous_capacity)
            split_data[split_type]["capacity_difference"].extend(capacity_difference)
            split_data[split_type]["cumulative_time_at_ground_truth"].extend(cumulative_time_at_ground_truth)

    def _convert_to_list(self, data: Any) -> List:
        """
        Convert data (which could be a dict or ndarray) to a list.

        :param data: Data to be converted.
        :return: List representation of the data.
        """
        if isinstance(data, dict):
            return list(data.values())
        elif isinstance(data, np.ndarray):
            return data.tolist()
        return data  # Assume it's already a list if not dict or ndarray

    def shuffle_and_split_graph_data(self, graph_data):
        """
        Shuffle and split graph data into training and validation sets deterministically.

        :param graph_data: The graph data structured by cell type, cell name, and cycle number.
        :return: A tuple containing train graphs and validation graphs.
        """

        train_graphs = []
        val_graphs = []
        val_split_ratio = self.config['model_specifications']['val_train_split_ratio']

        for cell_name, cycles in graph_data['train_cells'].items():
            # Convert to a list of tuples (cycle_no, graph)
            cycles_list = list(cycles.items())

            # Shuffle the cycles deterministically
            shuffled_indices = rng.permutation(len(cycles_list))
            cycles_list = [cycles_list[i] for i in shuffled_indices]

            # Determine split index
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

    # def prepare_graph_data(self, preprocessed_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    #     """
    #     Prepare the data for training and testing by constructing graph data for each cycle.
    #
    #     :param preprocessed_data: The preprocessed data structured as a dictionary.
    #     :return: A dictionary containing graph data for training and testing.
    #     """
    #
    #     cycle_graph_data = {'train_cells': {}, 'test_cells': {}}
    #
    #     for cell_type, cells in preprocessed_data.items():
    #         for cell_name, cell_data in cells.items():
    #             print(f"Preparing graph data for {cell_type}/{cell_name}")
    #
    #             previous_cycle_capacity = self.config['nominal_capacity']
    #
    #             cycle_graph_data[cell_type][cell_name] = {}
    #
    #             for cycle_no, cycle_data in cell_data.items():
    #                 graph = self.construct_graph_for_cycle(cycle_data, previous_cycle_capacity)
    #                 cycle_graph_data[cell_type][cell_name][cycle_no] = graph
    #
    #                 # Update previous_cycle_capacity for the next iteration
    #                 previous_cycle_capacity = cycle_data['ground_truth_cycle']['capacity']
    #
    #     return cycle_graph_data
    #
    # def construct_graph_for_cycle(self, data_point: Dict[str, Any], previous_cycle_capacity: float) -> Data:
    #     """
    #     Construct a graph from the provided data point.
    #
    #     :param data_point: A dictionary containing accumulated time bins, transitions, and ground truth cycle data.
    #     :param previous_cycle_capacity: The capacity from the previous cycle used as a global feature.
    #     :return: A PyTorch Geometric Data object representing the graph.
    #     """
    #
    #     # Nodes and their features (accumulated time in bins)
    #     node_features = []
    #     node_indices = []
    #
    #     for node, time_spent in data_point['accumulated_time_bins'].items():
    #         node_features.append([time_spent])
    #         node_indices.append(node)
    #
    #     node_features = torch.tensor(node_features, dtype=torch.float)
    #
    #     # Edges and their features (number of transitions)
    #     edge_indices = []
    #     edge_features = []
    #
    #     for (node_from, node_to), num_jumps in data_point['accumulated_transitions'].items():
    #         edge_indices.append([node_indices.index(node_from), node_indices.index(node_to)])
    #         edge_features.append([num_jumps])
    #
    #     edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(edge_features, dtype=torch.float)
    #
    #     # Global feature: previous cycle's capacity
    #     global_feat = torch.tensor([previous_cycle_capacity], dtype=torch.float)
    #
    #     # Target: capacity decay
    #     current_cycle_capacity = data_point['ground_truth_cycle']['capacity']
    #     capacity_decay = current_cycle_capacity - previous_cycle_capacity
    #     y = torch.tensor([capacity_decay], dtype=torch.float)
    #
    #     # Create and return the Data object
    #     return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y, global_feat=global_feat)

    def get_bin_mapping(self):
        """
        Create a mapping from flat indices to bin indices for voltage, current, and temperature.

        :return: A dictionary where the keys are flat indices and the values are tuples of bin indices.
        """
        bin_mapping = {}
        for i in range(len(self.voltage_bins)):
            for j in range(len(self.current_bins)):
                for k in range(len(self.temperature_bins)):
                    flat_idx = i * self.num_bins_per_dim[1] * self.num_bins_per_dim[2] + j * self.num_bins_per_dim[
                        2] + k
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
