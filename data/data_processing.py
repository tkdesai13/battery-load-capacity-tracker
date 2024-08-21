# Description: This file contains the DataProcessor class, which is responsible for loading, preprocessing, and
# structuring the data for training and testing the model.

import os
import itertools
from datetime import datetime
import scipy.io
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Tuple


def matstruct_to_dict(matstruct: Any) -> dict:
    """
    Recursively converts a MATLAB struct (loaded as mat_struct) to a Python dictionary.

    :param matstruct: The MATLAB struct to convert.
    :return: The converted Python dictionary.
    """
    result = {}
    for field_name in matstruct._fieldnames:
        field_value = getattr(matstruct, field_name)
        if isinstance(field_value, scipy.io.matlab.mio5_params.mat_struct):
            result[field_name] = matstruct_to_dict(field_value)
        elif isinstance(field_value, np.ndarray):
            result[field_name] = matstruct_array_to_list(field_value)
        else:
            result[field_name] = field_value
    return result


def matstruct_array_to_list(ndarray: np.ndarray) -> list:
    """
    Converts a MATLAB struct array into a list of Python dictionaries or returns the ndarray
    as a list if it is not a struct.

    :param ndarray: The numpy array to convert.
    :return: The converted list.
    """
    if ndarray.dtype == 'O':  # object type, usually for structs
        return [matstruct_to_dict(elem) for elem in ndarray]
    return ndarray.tolist()


class DataProcessor:
    def __init__(self, config: dict):
        self.config = config

    def load_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load the data from the data path and structure it into a dictionary.

        :return: The raw data structured as a dictionary.
        """
        data_path = self.config['paths']['data']
        raw_data = {}

        for cell_type, cells in self.config['cells'].items():
            raw_data[cell_type] = {}
            for cell in cells:
                cell_name = str(cell)
                print(f"Loading data for {cell_type}/{cell_name}")
                file_path = self._construct_file_path(data_path, cell_name)
                raw_data[cell_type][cell_name] = self._load_and_process_file(file_path)

        return raw_data

    def _construct_file_path(self, base_path: str, cell_name: str) -> str:
        """
        Construct the file path for the .mat file.

        :param base_path: The base path for the data.
        :param cell_name: The name of the cell.
        :return: The full file path.
        """
        return os.path.join(base_path, f"{cell_name}.mat")

    def _load_and_process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load and process the .mat file, converting it to a dictionary.

        :param file_path: The path to the .mat file.
        :return: The processed data as a dictionary.
        """
        try:
            mat_data = scipy.io.loadmat(file_name=file_path, squeeze_me=True, struct_as_record=False)
            return matstruct_to_dict(mat_data.get('data', {}))
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None

    def preprocess_data(self, raw_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Preprocess the raw data.

        :param raw_data: The raw data structured as a dictionary.
        :return: The preprocessed data structured as a dictionary.
        """
        preprocessed_data = {}

        for cell_type, cells in raw_data.items():
            preprocessed_data[cell_type] = {}
            for cell_name, cell_data in cells.items():
                print(f"Preprocessing data for {cell_type}/{cell_name}")
                preprocessed_data[cell_type][cell_name] = self._preprocess_cell_data(cell_data)

        return preprocessed_data

    def _preprocess_cell_data(self, cell_data: Dict[str, Any]) -> dict[int, dict[Any, Any]]:
        """
        Preprocess the data for a single cell.

        :param cell_data: The raw data for a single cell.
        :return: The preprocessed data for the cell.
        """
        ground_truth_cycle_counter = 1
        processed_data_stepwise = {}

        # Initialize a dictionary to accumulate time spent and transitions
        accumulated_time_bins = defaultdict(float)
        accumulated_transitions = defaultdict(int)

        for step_counter, step_data in enumerate(cell_data['step']):

            # temporary break for testing
            # if step_counter >= 10000:
            #     break

            # Skip this step if the time data has only one value
            relative_time = step_data['relativeTime']
            if isinstance(relative_time, (float, int)) or (isinstance(relative_time, list) and len(relative_time) <= 1):
                print(f"Skipping step {step_counter + 1} due to insufficient time data.")
                continue

            # Initialize the ground_truth_cycle_counter key if not already initialized
            if ground_truth_cycle_counter not in processed_data_stepwise:
                processed_data_stepwise[ground_truth_cycle_counter] = {}

            # Process the load pattern for each step and store it in time bins and transition matrix(jumps among bins)
            processed_data = self._process_load_pattern_each_step(step_data)

            # Add the processed time_bins to the accumulated dictionary
            accumulated_time_bins = self._add_time_bins(accumulated_time_bins, processed_data["time_bins"])

            # Update transition matrix
            accumulated_transitions = self._update_transition_matrix(accumulated_transitions, processed_data["transitions"])

            # Store the processed data in the dictionary
            processed_data_stepwise[ground_truth_cycle_counter][step_counter] = processed_data

            # Check for the reference discharge step to process capacity data
            if step_data['comment'] == 'reference discharge':
                processed_data_stepwise[ground_truth_cycle_counter][
                    'ground_truth_cycle'] = self._process_reference_capacity_step(step_data)
                processed_data_stepwise[ground_truth_cycle_counter]['accumulated_time_bins'] = dict(accumulated_time_bins)
                processed_data_stepwise[ground_truth_cycle_counter]['accumulated_transitions'] = dict(accumulated_transitions)

                # Reset the dictionaries for the next cycle
                accumulated_time_bins = defaultdict(float)
                accumulated_transitions = defaultdict(int)

                ground_truth_cycle_counter += 1

        return processed_data_stepwise

    def _process_load_pattern_each_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the load pattern data by calculating the time spent in each bin
        and the transitions between bins based on voltage, current (positive/negative), and temperature.

        :param step_data: The data for a single step.
        :return: The processed load pattern data.
        """
        # Extract the relevant data arrays
        voltage_data = step_data["voltage"]
        current_data = step_data["current"]
        temperature_data = step_data["temperature"]
        time_data = step_data["relativeTime"]  # This is the relative step time in seconds starting from 0 for each step

        # Bin the data using the vectorized approach
        time_bins, transitions = self.bin_battery_data_vectorized(voltage_data, current_data, temperature_data, time_data)

        # Store the processed data
        processed_data = {
            "time_bins": time_bins,
            "transitions": transitions  # Store transitions if needed for further analysis
        }

        return processed_data


    def bin_battery_data_vectorized(self, voltage, current, temperature, time):
        """
        Bin the battery data and calculate the time spent in each bin and transitions between bins.
        This version uses continuous current bins ranging from negative to positive.

        :param voltage: The voltage data.
        :param current: The current data.
        :param temperature: The temperature data.
        :param time: The time data.
        :return: The time spent in each bin and transitions between bins.
        """
        # Retrieve bins from the config
        voltage_bins = np.array(self.config['time_bins']['voltage_bins'])
        current_bins = np.array(self.config['time_bins']['current_bins'])  # Combined negative and positive bins
        temperature_bins = np.array(self.config['time_bins']['temperature_bins'])

        def digitize_data(data, bins):
            # Handle data outside the bins range
            indices = np.digitize(data, bins[:, 1], right=True)
            indices = np.clip(indices, 1, len(bins))  # Clip to ensure it stays within valid bin ranges
            return indices

        v_indices = digitize_data(voltage, voltage_bins)
        c_indices = digitize_data(current, current_bins)
        t_indices = digitize_data(temperature, temperature_bins)

        combined_indices = np.stack((v_indices, c_indices, t_indices), axis=-1)
        time_diffs = np.diff(time)

        # Initialize time_bins with all possible combinations of bins
        time_bins = defaultdict(float)
        for v_bin, c_bin, t_bin in itertools.product(range(1, len(voltage_bins) + 1),
                                                     range(1, len(current_bins) + 1),
                                                     range(1, len(temperature_bins) + 1)):
            time_bins[(v_bin, c_bin, t_bin)] = 0.0

        # Calculate time spent in each bin
        for idx, t in zip(combined_indices[1:], time_diffs):
            time_bins[tuple(idx)] += t / 3600  # Convert seconds to hours

        # Initialize transition matrix
        transitions = defaultdict(int)
        prev_indices = combined_indices[:-1]
        curr_indices = combined_indices[1:]
        changed_mask = np.any(prev_indices != curr_indices, axis=1)

        # Calculate transitions between bins
        for prev, curr in zip(prev_indices[changed_mask], curr_indices[changed_mask]):
            transitions[tuple(prev), tuple(curr)] += 1

        return dict(time_bins), dict(transitions)

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

    def _process_reference_capacity_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the capacity measurement data.

        :param step_data: The data for a single step.
        :return: The processed capacity measurement data.
        """
        # Convert the date string to a datetime object
        date_str = step_data.get("date", "")
        step_date = datetime.strptime(date_str, "%d-%b-%Y %H:%M:%S") if date_str else None

        # Extract current and relative time data
        current_data = step_data["current"]
        relative_time_data = step_data["relativeTime"]

        # Convert relative time from seconds to hours
        relative_time_hours = np.array(relative_time_data) / 3600.0

        # Calculate capacity using numerical integration (similar to MATLAB's trapz)
        capacity = np.trapz(current_data, relative_time_hours)

        # Store the processed data
        processed_data = {
            "date": step_date,
            "capacity": capacity
        }

        return processed_data

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
                train_test_data[cell_type][cell_name] = {"features": [], "target": []}

                # Extract the features for each cycle
                features = []
                for cycle_no, cycle_data in cell_data.items():
                    if 'accumulated_time_bins' in cycle_data:
                        accumulated_time_bins = list(cycle_data['accumulated_time_bins'].values())
                        features.append(accumulated_time_bins)
                    else:
                        print(
                            f"Warning: 'accumulated_time_bins' not found for {cell_type}/{cell_name} at cycle {cycle_no}")

                # Calculate delta_Q for target variable
                delta_q = []
                for cycle_no in cell_data.keys():
                    if cycle_no == 1:
                        # First cycle delta_Q is based on nominal capacity
                        if 'ground_truth_cycle' in cell_data[cycle_no]:
                            initial_capacity = cell_data[cycle_no]['ground_truth_cycle']['capacity']
                            delta_q.append(initial_capacity - self.config['nominal_capacity'])
                        else:
                            print(
                                f"Warning: 'ground_truth_cycle' not found for {cell_type}/{cell_name} at cycle {cycle_no}")
                    else:
                        # Subsequent delta_Q based on difference with previous cycle's capacity
                        if 'ground_truth_cycle' in cell_data[cycle_no] and 'ground_truth_cycle' in cell_data[cycle_no - 1]:
                            current_capacity = cell_data[cycle_no]['ground_truth_cycle']['capacity']
                            previous_capacity = cell_data[cycle_no - 1]['ground_truth_cycle']['capacity']
                            delta_q.append(current_capacity - previous_capacity)
                        else:
                            print(
                                f"Warning: 'ground_truth_cycle' not found for {cell_type}/{cell_name} at cycle {cycle_no}")

                # Store the features and target variables in the dictionary
                train_test_data[cell_type][cell_name]["features"] = features
                train_test_data[cell_type][cell_name]["target"] = delta_q

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
                "target": []
            },
            "validation_cells": {
                "features": [],
                "target": []
            },
            "test_cells": {}
        }

        # Combine and shuffle training data
        combined_features = []
        combined_target = []

        for cell_name, data in train_test_data["train_cells"].items():
            combined_features.extend(data["features"])
            combined_target.extend(data["target"])

        # Shuffle the combined features and targets together
        combined = list(zip(combined_features, combined_target))
        np.random.shuffle(combined)
        shuffled_features, shuffled_target = zip(*combined)

        # Calculate the split index for validation
        val_split_ratio = self.config['model_specifications']['val_train_split_ratio']
        split_index = int(len(shuffled_features) * (1 - val_split_ratio))

        # Split the data into training and validation sets
        split_data["train_cells"]["features"] = list(shuffled_features[:split_index])
        split_data["train_cells"]["target"] = list(shuffled_target[:split_index])
        split_data["validation_cells"]["features"] = list(shuffled_features[split_index:])
        split_data["validation_cells"]["target"] = list(shuffled_target[split_index:])

        # Retain test data without shuffling
        split_data["test_cells"] = train_test_data["test_cells"]

        return split_data

