import os
import h5py
import pickle
import numpy as np

# File paths to the .mat files for each batch
FILE_PATHS = {
    'b1': '/Users/tkddesai/surfdrive/Main drive/Data/Severson_and_Attia/2017-05-12_batchdata_updated_struct_errorcorrect.mat',
    'b2': '/Users/tkddesai/surfdrive/Main drive/Data/Severson_and_Attia/2017-06-30_batchdata_updated_struct_errorcorrect.mat',
    'b3': '/Users/tkddesai/surfdrive/Main drive/Data/Severson_and_Attia/2018-04-12_batchdata_updated_struct_errorcorrect.mat'
}

# Directory where the .pkl files will be saved
DATA_PATH = '/Users/tkddesai/surfdrive/Main drive/Data/Severson_and_Attia/'


def load_mat_file(file_path, batch_name):
    """
    Loads a .mat file and saves its contents as a .pkl file.

    Parameters:
    - file_path: Path to the .mat file.
    - batch_name: Name of the batch (e.g., 'b1', 'b2', 'b3').

    Returns:
    - bat_dict: A dictionary containing battery data from the .mat file.
    """
    bat_dict = {}

    # Open the .mat file using h5py
    with h5py.File(file_path, 'r') as f:
        batch = f['batch']
        num_cells = batch['summary'].shape[0]

        for i in range(num_cells):
            # Extract data for each cell
            cycle_life = f[batch['cycle_life'][i, 0]][()]
            policy = f[batch['policy_readable'][i, 0]][()].tobytes()[::2].decode()

            summary_data = f[batch['summary'][i, 0]]
            summary = {
                'IR': summary_data['IR'][0, :].tolist(),
                'QC': summary_data['QCharge'][0, :].tolist(),
                'QD': summary_data['QDischarge'][0, :].tolist(),
                'Tavg': summary_data['Tavg'][0, :].tolist(),
                'Tmin': summary_data['Tmin'][0, :].tolist(),
                'Tmax': summary_data['Tmax'][0, :].tolist(),
                'chargetime': summary_data['chargetime'][0, :].tolist(),
                'cycle': summary_data['cycle'][0, :].tolist()
            }

            # Load cycle data
            cycles = f[batch['cycles'][i, 0]]
            num_cycles = cycles['I'].shape[0]
            cycle_dict = {}

            # Fields to extract from cycles
            fields = ['I', 'Qc', 'Qd', 'Qdlin', 'T', 'Tdlin', 'V', 'discharge_dQdV', 't']

            # Load data for all fields
            field_data = {}
            for field in fields:
                refs = cycles[field][:, 0]
                field_data[field] = [f[ref][()] for ref in refs]

            # Assemble the cycle_dict
            for j in range(num_cycles):
                cycle_data = {field: field_data[field][j] for field in fields}
                cycle_dict[str(j)] = cycle_data

            # Assemble cell data
            cell_dict = {
                'cycle_life': cycle_life,
                'charge_policy': policy,
                'summary': summary,
                'cycles': cycle_dict
            }

            key = f'{batch_name}c{i}'
            bat_dict[key] = cell_dict
            print(f"Processed cell: {key}")

    # Save the bat_dict to a .pkl file
    output_file_name = f'batch_{batch_name}.pkl'
    output_file_path = os.path.join(DATA_PATH, output_file_name)

    try:
        with open(output_file_path, 'wb') as fp:
            pickle.dump(bat_dict, fp)
        print(f"Saved {output_file_name} successfully at {output_file_path}.")
    except IOError as e:
        print(f"Error saving the file {output_file_path}: {e}")

    return bat_dict


def process_battery_data(data_path):
    """
    Processes saved battery data by removing specified entries, updating others,
    and combining all batches into a single dictionary. Saves the modified data
    as a new pickle file apart from the original files.

    Parameters:
    - data_path: Directory path where the original batch .pkl files are located.

    Returns:
    - combined_battery_data: The combined and processed battery data dictionary.
    """
    # Load the original batch data
    batch_files = {
        'batch1': os.path.join(data_path, 'batch_b1.pkl'),
        'batch2': os.path.join(data_path, 'batch_b2.pkl'),
        'batch3': os.path.join(data_path, 'batch_b3.pkl')
    }

    with open(batch_files['batch1'], 'rb') as f:
        batch1 = pickle.load(f)
    with open(batch_files['batch2'], 'rb') as f:
        batch2 = pickle.load(f)
    with open(batch_files['batch3'], 'rb') as f:
        batch3 = pickle.load(f)

    # Batch 1 Modifications
    batteries_to_remove_b1 = ['b1c8', 'b1c10', 'b1c12', 'b1c13', 'b1c22']
    for battery in batteries_to_remove_b1:
        batch1.pop(battery, None)
        print(f"Removed {battery} from batch1.")

    num_batteries_b1 = len(batch1)
    print(f"Number of batteries in batch1 after removal: {num_batteries_b1}")

    # Batch 2 Modifications
    overlapping_batteries_b2 = ['b2c7', 'b2c8', 'b2c9', 'b2c15', 'b2c16']
    corresponding_batteries_b1 = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4']
    additional_cycles = [662, 981, 1060, 208, 482]

    for i, b1_key in enumerate(corresponding_batteries_b1):
        b2_key = overlapping_batteries_b2[i]
        # Update cycle life
        batch1[b1_key]['cycle_life'] += additional_cycles[i]

        # Update summary data
        for key in batch1[b1_key]['summary'].keys():
            b1_summary = np.array(batch1[b1_key]['summary'][key])
            b2_summary = np.array(batch2[b2_key]['summary'][key])

            if key == 'cycle':
                b2_summary += len(b1_summary)
            combined_summary = np.hstack((b1_summary, b2_summary))
            batch1[b1_key]['summary'][key] = combined_summary.tolist()

        # Update cycles data
        last_cycle = len(batch1[b1_key]['cycles'])
        for j, cycle_key in enumerate(batch2[b2_key]['cycles'].keys()):
            batch1[b1_key]['cycles'][str(last_cycle + j)] = batch2[b2_key]['cycles'][cycle_key]

        print(f"Updated {b1_key} with data from {b2_key}.")

    # Remove overlapping cells from batch2
    for battery in overlapping_batteries_b2:
        batch2.pop(battery, None)
        print(f"Removed {battery} from batch2.")

    num_batteries_b2 = len(batch2)
    print(f"Number of batteries in batch2 after removal: {num_batteries_b2}")

    # Batch 3 Modifications
    # remove noisy channels from batch3
    batteries_to_remove_b3 = ['b3c37', 'b3c2', 'b3c23', 'b3c32', 'b3c42', 'b3c43']
    for battery in batteries_to_remove_b3:
        batch3.pop(battery, None)
        print(f"Removed {battery} from batch3.")

    num_batteries_b3 = len(batch3)
    print(f"Number of batteries in batch3 after removal: {num_batteries_b3}")

    # Combine all batches
    combined_battery_data = {**batch1, **batch2, **batch3}
    total_batteries = len(combined_battery_data)
    print(f"Total number of batteries after combining: {total_batteries}")

    # Save the combined dictionary
    output_file = os.path.join(data_path, 'severson_processed_battery_data.pkl')
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(combined_battery_data, f)
        print(f"Saved processed battery data to {output_file}.")
    except IOError as e:
        print(f"Error saving the file {output_file}: {e}")

    return combined_battery_data


# Process each batch file and save the corresponding .pkl file
for batch_name, file_path in FILE_PATHS.items():
    load_mat_file(file_path, batch_name)

# Process the saved .pkl files to create the combined and cleaned data
bat_dict = process_battery_data(DATA_PATH)
