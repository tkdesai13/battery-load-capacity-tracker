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

