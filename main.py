import json
import warnings

from data_processing import Data_Process

# Ignore warnings
warnings.filterwarnings("ignore")

# Open the JSON Configuration file
with open('config.json', 'r') as json_file:
    config_data = json.load(json_file)
model_specs = config_data['model_specifications']

# Construct the path for experiment-specific results
experiment_result_dir = os.path.join(config_data['result_path'], config_data['experiment_name'])

# Ensure the directory exists, and create it if it doesn't
if not os.path.exists(experiment_result_dir):
    os.makedirs(experiment_result_dir)

# data Extraction for Model training
print("Start: Data extraction and Processing")
data_processing = Data_Process(config_data=config_data)

