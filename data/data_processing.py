import os
import scipy.io
import numpy as np
from typing import Dict, Any

def matstruct_to_dict(matstruct: Any) -> dict:
    """
    Recursively converts a MATLAB struct (loaded as mat_struct) to a Python dictionary.

    :param matstruct: The MATLAB struct to convert.
    :return: A dictionary representing the MATLAB struct.
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
    Converts a MATLAB struct array into a list of Python dictionaries or returns the ndarray.

    :param ndarray: The NumPy array to convert.
    :return: A list of dictionaries if the ndarray contains MATLAB structs, otherwise the ndarray.
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

        :return: A dictionary containing the raw data of all cells.
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

        :param base_path: The base directory path.
        :param cell_name: The name of the cell.
        :return: The full path to the .mat file.
        """
        return os.path.join(base_path, f"{cell_name}.mat")

    def _load_and_process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load and process the .mat file, converting it to a dictionary.

        :param file_path: The path to the .mat file.
        :return: A dictionary containing the processed data, or None if the file is not found.
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

        :param raw_data: The raw data to preprocess.
        :return: The preprocessed data.
        """
        preprocessed_data = {}

        for cell_type, cells in raw_data.items():
            preprocessed_data[cell_type] = {}
            for cell_name, cell_data in cells.items():
                print(f"Preprocessing data for {cell_type}/{cell_name}")
                preprocessed_data[cell_type][cell_name] = self._preprocess_cell_data(cell_data)

        return preprocessed_data

    def _preprocess_cell_data(self, cell_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess the data for a single cell.

        :param cell_data: The data for a single cell.
        :return: The preprocessed data.
        """
        ground_truth_cycle_counter = 0
        processed_data_stepwise = {}

        for step_counter, step_data in enumerate(cell_data['step']):

            processed_data_stepwise[ground_truth_cycle_counter][step_counter] = self._process_load_pattern_each_step(step_data)

            if step_data['comment'] == 'reference discharge':
                processed_data_stepwise[ground_truth_cycle_counter]['ground_truth_cycle'] = self._process_reference_capacity_step(step_data)
                ground_truth_cycle_counter += 1

        return cell_data

    def _process_load_pattern_each_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the load pattern data by calculating the time spent in each bin
        based on voltage, current (positive/negative), and temperature.

        :param step_data: The data for a single step.
        :return: The processed load pattern data with time spent in each bin.
        """
        # Access the bins from the configuration
        voltage_bins = self.config["time_bins"]["voltage_bins"]
        current_bins_positive = self.config["time_bins"]["current_bins"]["positive"]
        current_bins_negative = self.config["time_bins"]["current_bins"]["negative"]
        temperature_bins = self.config["time_bins"]["temperature_bins"]

        # Initialize the 3x3x2 matrix to store the time spent in each bin
        time_bins = [[[0 for _ in range(2)] for _ in range(3)] for _ in range(3)]

        # Extract the relevant data arrays
        voltage_data = step_data["voltage"]
        current_data = step_data["current"]
        temperature_data = step_data["temperature"]
        time_data = step_data["time"]  # Assuming this is the cumulative time in seconds

        # Calculate the time spent in each bin
        for i in range(1, len(voltage_data)):
            voltage = voltage_data[i]
            current = current_data[i]
            temperature = temperature_data[i]
            time_spent = time_data[i] - time_data[i - 1]

            # Determine which bin the data point falls into
            for v_idx, (v_min, v_max) in enumerate(voltage_bins):
                if v_min <= voltage < v_max:
                    if current >= 0:
                        for c_idx, (c_min, c_max) in enumerate(current_bins_positive):
                            if c_min <= current < c_max:
                                for t_idx, (t_min, t_max) in enumerate(temperature_bins):
                                    if t_min <= temperature < t_max:
                                        time_bins[v_idx][c_idx][0] += time_spent
                    else:
                        for c_idx, (c_min, c_max) in enumerate(current_bins_negative):
                            if c_min <= current < c_max:
                                for t_idx, (t_min, t_max) in enumerate(temperature_bins):
                                    if t_min <= temperature < t_max:
                                        time_bins[v_idx][c_idx][1] += time_spent

        # Store the processed data
        processed_data = {
            "time_bins": time_bins
        }

        return processed_data

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