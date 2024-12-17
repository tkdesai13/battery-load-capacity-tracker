import os
import json
import tensorflow as tf
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
import numpy as np
import pickle
from typing import Dict, Any, List
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, Span
from models.model import Model

class ModelEvaluator:
    def __init__(self, config: Dict[str, Any], model_instance: Model):
        self.config = config
        self.model_instance = model_instance
        self.model_dir = os.path.join(config['paths']['result'], config['experiment_name'])
        os.makedirs(self.model_dir, exist_ok=True)

    def evaluate_model(self, model, data: Dict[str, Any], min_max_values: Dict[str, Any],
                       max_sequence_length: int = None, test_cells: List[str] = None):
        """
        Evaluate the model using the provided training and testing data.

        :param model: A trained model.
        :param max_sequence_length: The maximum sequence length for the vector model.
        :param data: A dictionary containing the features and targets for training and testing cells.
        :param test_cells: List of cell IDs that are designated as test cells.
        """
        # Aggregated predictions for train and test splits
        aggregated_predictions = {
            'train': {'q_actual': [], 'q_predict': [], 'dq_actual': [], 'dq_predict': [], 'cycle_no': []},
            'test': {'q_actual': [], 'q_predict': [], 'dq_actual': [], 'dq_predict': [], 'cycle_no': []}
        }

        # Split the cells into train and test based on the test_cells list
        cell_types = {
            'train': [cell for batch in data.keys() for cell in data[batch] if cell not in test_cells],
            # All other cells are treated as train
            'test': test_cells,  # Use the provided test cells list
        }

        # Iterate over train and test cells and accumulate predictions
        for split_type, split_cells in cell_types.items():
            predictions = {}
            cell_metrics = {}

            # Iterate through the data batches and process each cell
            for batch in data.keys():
                for cell in data[batch].keys():
                    if cell in split_cells:
                        cell_data = data[batch][cell]

                        # Evaluate the model for this single cell
                        predictions[cell], cell_metrics[cell] = self._evaluate_single_cell(model, cell_data, min_max_values,
                                                                                     max_sequence_length)

                        # Combine predictions and metrics for the individual cell
                        combined_data = {
                            'metrics': cell_metrics[cell],
                            'predictions': predictions[cell]

                        }

                        # Accumulate actual, predicted values, dq values, and dates
                        aggregated_predictions[split_type]['q_actual'].extend(predictions[cell]['q_actual'])
                        aggregated_predictions[split_type]['q_predict'].extend(predictions[cell]['q_predict'])
                        aggregated_predictions[split_type]['dq_actual'].extend(predictions[cell]['dq_actual'])
                        aggregated_predictions[split_type]['dq_predict'].extend(predictions[cell]['dq_predict'])
                        aggregated_predictions[split_type]['cycle_no'].extend(predictions[cell]['cycle_no'])

                        # Save combined predictions and metrics for the individual cell
                        self._save_metrics(combined_data, f'{split_type}_{cell}_predictions.json')

            # Calculate aggregated metrics over all cells for this split type (train/test)
            metrics = self._calculate_aggregated_metrics(aggregated_predictions[split_type])

            # Save the overall metrics for the split type
            with open(os.path.join(self.model_dir, f'{split_type}_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)

            # Plot the results for this split type
            self._plot_results(predictions, cell_metrics, f'{split_type}_results.html')

        return aggregated_predictions

    def evaluate_sequence_to_sequence_model(self, model, train_val_test_data: Dict[str, Dict[str, Any]],
                                            max_encoder_timesteps: int, max_decoder_timesteps: int,
                                            max_transitions: int, output_filename_prefix: str = 'seq2seq_results'):
        """
        Evaluate the sequence-to-sequence model where each cell has multiple sequences.
        This version also plots encoder data and uses per-sequence metrics for plot titles.
        """

        aggregated_predictions = {
            'train': {'q_actual': [], 'q_predict': [], 'dq_actual': [], 'dq_predict': [], 'cycle_no': []},
            'validation': {'q_actual': [], 'q_predict': [], 'dq_actual': [], 'dq_predict': [], 'cycle_no': []},
            'test': {'q_actual': [], 'q_predict': [], 'dq_actual': [], 'dq_predict': [], 'cycle_no': []}
        }

        split_cell_predictions = {'train': {}, 'validation': {}, 'test': {}}

        for split_type in ['train', 'validation', 'test']:
            for cell_name, cell_data in train_val_test_data[split_type].items():
                num_sequences = int(self.config["model_specifications"]["max_no_of_sequences_per_cell"])

                # Pad and mask all sequences at once using pad_and_mask_data
                inputs, targets = self.model_instance.pad_and_mask_data(
                    cell_data, max_encoder_timesteps, max_decoder_timesteps, max_transitions, pre_pad_encoder=False
                )
                # predictions: (num_sequences, max_decoder_timesteps, 1)
                predictions = model.predict(inputs, verbose=0)
                dq_pred_all = predictions[:, :, 0]  # (num_sequences, max_decoder_timesteps)
                dq_true_all = targets[:, :, 0]  # (num_sequences, max_decoder_timesteps)

                cell_sequences_predictions = []
                cell_q_actual_all, cell_q_predict_all = [], []
                cell_dq_actual_all, cell_dq_predict_all = [], []
                cell_cycle_no_all = []

                for seq_idx in range(num_sequences):
                    dq_pred = dq_pred_all[seq_idx]
                    dq_true = dq_true_all[seq_idx]

                    # Extract initial_q from the last encoder timestep capacity
                    initial_q = cell_data['decoder_inputs']['previous_capacity'][seq_idx][0]

                    # Get cycle_no for decoder from decoder_inputs['cycle_no'][seq_idx]
                    cycle_no = cell_data['decoder_inputs']['cycle_no'][seq_idx]

                    # Reconstruct Q
                    q_pred = np.cumsum(dq_pred) + initial_q
                    q_true = cell_data['decoder_inputs']['capacity'][seq_idx]

                    # Sequence length without padding
                    decoder_sequence_length_without_padding = len(cycle_no)
                    q_pred = q_pred[:decoder_sequence_length_without_padding].tolist()
                    dq_true = dq_true[:decoder_sequence_length_without_padding].tolist()
                    dq_pred = dq_pred[:decoder_sequence_length_without_padding].tolist()

                    # Compute per-sequence metrics
                    seq_metrics = self._compute_metrics(q_true, q_pred, dq_true, dq_pred)

                    cell_sequences_predictions.append({
                        'cycle_no': cycle_no,
                        'q_actual': q_true,
                        'q_predict': q_pred,
                        'dq_actual': dq_true,
                        'dq_predict': dq_pred,
                        'metrics': seq_metrics
                    })

                    # Accumulate for cell-level metrics
                    cell_q_actual_all.extend(q_true)
                    cell_q_predict_all.extend(q_pred)
                    cell_dq_actual_all.extend(dq_true)
                    cell_dq_predict_all.extend(dq_pred)
                    cell_cycle_no_all.extend(cycle_no)

                # Compute cell-level metrics
                cell_metrics = self._compute_metrics(cell_q_actual_all, cell_q_predict_all, cell_dq_actual_all,
                                                     cell_dq_predict_all)

                # Update aggregated predictions
                aggregated_predictions[split_type]['q_actual'].extend(cell_q_actual_all)
                aggregated_predictions[split_type]['q_predict'].extend(cell_q_predict_all)
                aggregated_predictions[split_type]['dq_actual'].extend(cell_dq_actual_all)
                aggregated_predictions[split_type]['dq_predict'].extend(cell_dq_predict_all)
                aggregated_predictions[split_type]['cycle_no'].extend(cell_cycle_no_all)

                # Store cell predictions and metrics
                # Note: We still store cell-level metrics but now each sequence has its own metrics too
                split_cell_predictions[split_type][cell_name] = {
                    'sequences': cell_sequences_predictions,
                    'metrics': cell_metrics,
                    'encoder_inputs': cell_data['encoder_inputs']  # needed for plotting encoder part
                }

            # Compute overall metrics for this split
            split_metrics = self._compute_metrics(
                aggregated_predictions[split_type]['q_actual'],
                aggregated_predictions[split_type]['q_predict'],
                aggregated_predictions[split_type]['dq_actual'],
                aggregated_predictions[split_type]['dq_predict']
            )

            # Save overall metrics
            with open(os.path.join(self.model_dir, f'{split_type}_overall_metrics.json'), 'w') as f:
                json.dump(split_metrics, f, indent=4)

            # Plot results for this split
            self._plot_sequences_bokeh(split_type, split_cell_predictions[split_type],
                                       f'{output_filename_prefix}_{split_type}.html')

            # After all splits are processed, save *everything*:
            # This includes the original input data and all predictions/metrics
            all_data_to_save = {
                'train_val_test_data': train_val_test_data,
                'split_cell_predictions': split_cell_predictions
            }

            with open(os.path.join(self.model_dir, 'all_cell_data_with_predictions.pkl'), 'wb') as f:
                pickle.dump(all_data_to_save, f)

        return aggregated_predictions

    def _compute_metrics(self, q_actual, q_predict, dq_actual, dq_predict):
        # Convert lists to numpy arrays for arithmetic
        q_actual = np.array(q_actual)
        q_predict = np.array(q_predict)
        dq_actual = np.array(dq_actual)
        dq_predict = np.array(dq_predict)

        cell_metrics = {
            'q_metrics': {
                'rmse': np.sqrt(mean_squared_error(q_actual, q_predict)),
                'mae': mean_absolute_error(q_actual, q_predict),
                'mape': np.mean(np.abs((q_actual - q_predict) / q_actual)) * 100,
                'r2': r2_score(q_actual, q_predict),
                'mse': mean_squared_error(q_actual, q_predict)
            },
            'dq_metrics': {
                'rmse': np.sqrt(mean_squared_error(dq_actual, dq_predict)),
                'crmse': np.sqrt(np.mean(((dq_actual - dq_predict) / q_actual) ** 2)) * 100,
                'mae': mean_absolute_error(dq_actual, dq_predict),
                'mape': np.mean(np.abs((dq_actual - dq_predict) / q_actual)) * 100,
                'r2': r2_score(dq_actual, dq_predict),
                'mse': mean_squared_error(dq_actual, dq_predict)
            }
        }
        return cell_metrics

    def _plot_sequences_bokeh(self, split_type: str, cell_data: Dict[str, Any], output_filename: str):

        rows = []
        for cell, cell_info in cell_data.items():
            sequences = cell_info['sequences']
            encoder_data = cell_info['encoder_inputs']

            q_plots = []
            dq_plots = []

            # encoder_data is a dictionary with keys: ['bin_indices', 'timebins', 'capacity', 'previous_capacity', 'capacity_difference', 'cumulative_time', 'cycle_no']
            # Each entry of encoder_data['capacity'][seq_idx], encoder_data['cycle_no'][seq_idx], etc. corresponds to a single sequence's encoder portion.

            for seq_idx, seq_pred in enumerate(sequences):
                # Extract encoder portion for this sequence
                encoder_cycle_no = encoder_data['cycle_no'][seq_idx]
                encoder_q_actual = encoder_data['capacity'][seq_idx]
                encoder_dq_actual = encoder_data['capacity_difference'][seq_idx]

                # Create encoder source
                encoder_source = ColumnDataSource(data=dict(
                    x=encoder_cycle_no,
                    q_actual=encoder_q_actual,
                    dq_actual=encoder_dq_actual
                ))

                # Create decoder source
                source = ColumnDataSource(data=dict(
                    x=seq_pred['cycle_no'],
                    q_actual=seq_pred['q_actual'],
                    q_predict=seq_pred['q_predict'],
                    dq_actual=seq_pred['dq_actual'],
                    dq_predict=seq_pred['dq_predict']
                ))

                # Use per-sequence metrics for titles
                seq_metrics = seq_pred['metrics']

                p_q = figure(
                    title=(f"Cell {cell}, Q - "
                           f"RMSE:{seq_metrics['q_metrics']['rmse']:.4f}, "
                           f"MAPE:{seq_metrics['q_metrics']['mape']:.2f}%"),
                    x_axis_label='Cycle Number', y_axis_label='Q (Ah)', width=400, height=300)

                # Plot encoder actual Q
                p_q.line('x', 'q_actual', source=encoder_source, legend_label='Q actual', line_width=2,
                         color='blue')
                # Plot decoder actual and predicted Q
                p_q.line('x', 'q_actual', source=source, line_width=2, color='blue')
                p_q.line('x', 'q_predict', source=source, legend_label='Q predict', line_width=2, color='red')

                # Add a vertical line at the boundary: last encoder cycle
                if len(encoder_cycle_no) > 0:
                    boundary = encoder_cycle_no[-1]
                    p_q.add_layout(Span(location=boundary, dimension='height', line_color='black', line_dash='dashed',
                                        line_width=2))

                p_q.legend.location = 'top_left'

                p_dq = figure(
                    title=(f"Cell {cell}, dQ - "
                           f"RMSE:{seq_metrics['dq_metrics']['rmse']:.4f}, "
                           f"MAPE:{seq_metrics['dq_metrics']['mape']:.2f}%"),
                    x_axis_label='Cycle Number', y_axis_label='dQ (Ah)', width=400, height=300)

                # Plot encoder actual dQ
                p_dq.line('x', 'dq_actual', source=encoder_source, legend_label='dQ actual', line_width=2,
                          color='green')
                # Plot decoder actual and predicted dQ
                p_dq.line('x', 'dq_actual', source=source, line_width=2,
                          color='green')
                p_dq.line('x', 'dq_predict', source=source, legend_label='dQ predict', line_width=2, color='orange')

                if len(encoder_cycle_no) > 0:
                    p_dq.add_layout(Span(location=boundary, dimension='height', line_color='black', line_dash='dashed',
                                         line_width=2))

                p_dq.legend.location = 'top_left'

                q_plots.append(p_q)
                dq_plots.append(p_dq)

            cell_row = gridplot([q_plots, dq_plots])
            rows.append([cell_row])

        final_layout = gridplot(rows)
        output_file(os.path.join(self.model_dir, output_filename))
        save(final_layout)

    def _evaluate_single_cell(self, model, cell_data: Dict[str, Any], min_max_values: Dict, max_sequence_length: int):
        """
        Evaluate a single cell's performance.

        :param model: A trained model.
        :param cell_data: A dictionary containing the features, target, and ground truth cycles.
        :return predictions: A dictionary containing predictions, actual values, and dates.
        :return cell_metrics: A dictionary containing the error metrics for this cell.
        """
        predictions = {}

        previous_capacity_all_cycle_cells = np.array(
            [cycle_data['previous_capacity'] for cycle_no, cycle_data in cell_data.items()])

        cumulative_time_at_ground_truth_all_cells = np.array(
            [cycle_data['cumulative_time_at_ground_truth'] for cycle_no, cycle_data in cell_data.items()])

        # Ensure that previous_capacity_all_cycle_cells is correctly shaped as (N, 1) where N is the number of cycles
        previous_capacity_all_cycle_cells = previous_capacity_all_cycle_cells.reshape(-1, 1)
        cumulative_time_at_ground_truth = cumulative_time_at_ground_truth_all_cells.reshape(-1,1)

        # Extract time bins and previous_capacity for all cycles
        if self.config['scalar_vector_exp_flag'] == 'scalar':
            timebins_all_cycle_cells = np.array(
                [list(cycle_data['accumulated_time_bins'].values()) for cycle_no, cycle_data in cell_data.items()])

            # Predicted dq values using the model, pass both inputs in a list
            dq_predict = model.predict([timebins_all_cycle_cells, previous_capacity_all_cycle_cells]).flatten()

        elif self.config['scalar_vector_exp_flag'] == 'vector':
            timebins_all_cycle_cells = [list(cycle_data['accumulated_time_bins']) for cycle_no, cycle_data in cell_data.items()]

            # Pad the timebins sequences for both train and validation data
            padded_train_timebins = pad_sequences(timebins_all_cycle_cells, maxlen=max_sequence_length, padding='pre',
                                                  dtype='float32', value=0.0)

            # Predicted dq values using the model, pass both inputs in a list
            dq_predict = model.predict([tf.convert_to_tensor(padded_train_timebins),
                                        tf.convert_to_tensor(previous_capacity_all_cycle_cells)]).flatten()

        else:
            # Third case: Adjusted to align with model's inputs
            # Extract bin indices and time spent for all cycles
            bin_indices_all_cycle_cells = [list(cycle_data['bin_indices']) for cycle_no, cycle_data in cell_data.items()]

            time_spent_all_cycle_cells = [list(cycle_data['accumulated_time_bins']) for cycle_no, cycle_data in cell_data.items()]

            # Pad the bin_indices sequences
            padded_bin_indices = pad_sequences(
                bin_indices_all_cycle_cells,
                maxlen=max_sequence_length,
                padding='post',
                value=0  # Padding index for bin_indices
            )

            # Pad the time_spent sequences
            padded_time_spent = pad_sequences(
                time_spent_all_cycle_cells,
                maxlen=max_sequence_length,
                padding='post',
                value=0.0,  # Padding value for time_spent
                dtype='float32'
            )

            # Optionally, scale previous_capacity and time_spent if scaling was applied during training
            # Example (uncomment and adjust if necessary):
            # previous_capacity_all_cycle_cells = self.scaler_prev_capacity.transform(previous_capacity_all_cycle_cells)
            # padded_time_spent = self.scaler_time_spent.transform(padded_time_spent)

            # Prepare the inputs as dictionaries matching the model's input layer names
            inputs = {
                'bin_index_input': padded_bin_indices.astype('int32'),  # Shape: (batch_size, max_sequence_length)
                'time_spent_input': padded_time_spent.astype('float32'),  # Shape: (batch_size, max_sequence_length)
                'previous_capacity_input': previous_capacity_all_cycle_cells.astype('float32')  # Shape: (batch_size, 1)
            }

            # Predict dq values using the model
            dq_predict = model.predict(inputs).flatten()

        # Actual dq values
        dq_actual = np.array([cycle_data['capacity_diff'] for cycle_no, cycle_data in cell_data.items()])

        # Previous capacity values
        q_prev_step = np.array([cycle_data['previous_capacity'] for cycle_no, cycle_data in cell_data.items()])

        # Descaling the capacity difference and other features, used for plot
        if self.config['model_config']['data_scaling'] is True:
            if 'capacity_diff' in self.config['model_config']['signals_to_scale']:
                dq_predict = self.descale_value(dq_predict, min_max_values['capacity_diff'])
                dq_actual = self.descale_value(dq_actual, min_max_values['capacity_diff'])
            if 'previous_capacity' in self.config['model_config']['signals_to_scale']:
                q_prev_step = self.descale_value(q_prev_step, min_max_values['previous_capacity'])

        # Actual capacity values
        q_actual = np.array([cycle_data['capacity'] for cycle_no, cycle_data in cell_data.items()])
        cycle_no = np.array([cycle_data['cycle_no'] for cycle_no, cycle_data in cell_data.items()])

        q_predict = [q_prev + dq_current_predict for q_prev, dq_current_predict in zip(q_prev_step, dq_predict.tolist())]

        # Store predictions, actual values, dq values, and dates
        predictions['q_actual'] = q_actual.tolist()
        predictions['q_predict'] = q_predict
        predictions['dq_actual'] = dq_actual.tolist()
        predictions['dq_predict'] = dq_predict.tolist()
        predictions['cycle_no'] = cycle_no.tolist()

        # Individual cell metrics for q and dq
        cell_metrics = {
            'q_metrics': {
                'rmse': np.sqrt(mean_squared_error(q_actual, q_predict)),
                'mae': mean_absolute_error(q_actual, q_predict),
                'mape': np.mean(np.abs((q_actual - q_predict) / q_actual)) * 100,
                'r2': r2_score(q_actual, q_predict),
                'mse': mean_squared_error(q_actual, q_predict)
            },
            'dq_metrics': {
                'rmse': np.sqrt(mean_squared_error(dq_actual, dq_predict)),
                'crmse': np.sqrt(np.mean(((dq_actual - dq_predict) / q_actual) ** 2)) * 100,
                'mae': mean_absolute_error(dq_actual, dq_predict),
                'mape': np.mean(np.abs((dq_actual - dq_predict) / q_actual)) * 100,
                'r2': r2_score(dq_actual, dq_predict),
                'mse': mean_squared_error(dq_actual, dq_predict)
            }
        }

        return predictions, cell_metrics

    def _calculate_aggregated_metrics(self, aggregated_predictions: Dict[str, list]):
        """
        Calculate the aggregated metrics for all cells combined.

        :param aggregated_predictions: A dictionary containing combined predictions, actual values, dq values, and dates.
        :return: A dictionary containing the aggregated error metrics.
        """
        q_actual = np.array(aggregated_predictions['q_actual'])
        q_predict = np.array(aggregated_predictions['q_predict'])
        dq_actual = np.array(aggregated_predictions['dq_actual'])
        dq_predict = np.array(aggregated_predictions['dq_predict'])

        metrics = {
            'q_metrics': {
                'rmse': np.sqrt(mean_squared_error(q_actual, q_predict)),
                'mae': mean_absolute_error(q_actual, q_predict),
                'mape': np.mean(np.abs((q_actual - q_predict) / q_actual)) * 100,
                'r2': r2_score(q_actual, q_predict),
                'mse': mean_squared_error(q_actual, q_predict)
            },
            'dq_metrics': {
                'rmse': np.sqrt(mean_squared_error(dq_actual, dq_predict)),
                'crmse': np.sqrt(np.mean(((dq_actual - dq_predict) / q_actual) ** 2)) * 100,
                'mae': mean_absolute_error(dq_actual, dq_predict),
                'mape': np.mean(np.abs((dq_actual - dq_predict) / q_actual)) * 100,
                'r2': r2_score(dq_actual, dq_predict),
                'mse': mean_squared_error(dq_actual, dq_predict)
            }
        }

        return metrics

    def _save_metrics(self, predictions: Dict[str, Any], filename: str):
        """
        Save the predictions and metrics to a JSON file.

        :param predictions: The predictions dictionary to save.
        :param filename: The filename to save the predictions.
        """
        filepath = os.path.join(self.model_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(predictions, f, indent=4)

    def _plot_results(self, predictions: Dict[str, Any], cell_metrics: Dict[str, Any], output_filename: str):
        """
        Plot the predictions versus actual values using Bokeh and save the results.

        :param predictions: The predictions dictionary containing actual and predicted values along with dates.
        :param output_filename: The filename to save the HTML plot.
        """
        plots = []
        for cell, cell_predictions in predictions.items():

            # cell metrics
            cell_q_rmse = cell_metrics[cell]['q_metrics']['rmse']
            cell_q_mape = cell_metrics[cell]['q_metrics']['mape']
            cell_dq_rmse = cell_metrics[cell]['dq_metrics']['rmse']
            cell_dq_crmse = cell_metrics[cell]['dq_metrics']['crmse']
            cell_dq_mae = cell_metrics[cell]['dq_metrics']['mae']
            cell_dq_mape = cell_metrics[cell]['dq_metrics']['mape']

            source_q = ColumnDataSource(data=dict(
                x=cell_predictions['cycle_no'],
                y_actual=cell_predictions['q_actual'],
                y_predict=cell_predictions['q_predict']
            ))

            source_dq = ColumnDataSource(data=dict(
                x=cell_predictions['cycle_no'],
                y_actual=cell_predictions['dq_actual'],
                y_predict=cell_predictions['dq_predict']
            ))

            # Plot for capacity (q)
            p_q = figure(title=f'Cell {cell} - RMSE:{cell_q_rmse:.4f}, MAPE:{cell_q_mape:.4f}',
                         x_axis_label='Cyele Number',
                         y_axis_label='Capacity (Ah)',
                         width=400,
                         height=400)

            p_q.line('x', 'y_actual', source=source_q, legend_label='Actual Q', line_width=2, line_color='blue')
            p_q.line('x', 'y_predict', source=source_q, legend_label='Predicted Q', line_width=2, line_color='red')
            p_q.circle('x', 'y_actual', source=source_q, size=6, color='blue', legend_label='Actual Q', fill_alpha=0.6)
            p_q.circle('x', 'y_predict', source=source_q, size=6, color='red', legend_label='Predicted Q',
                       fill_alpha=0.6)
            p_q.legend.location = 'top_left'

            # Regression and Identity Line for q_actual vs q_predict
            max_q = max(max(cell_predictions['q_actual']), max(cell_predictions['q_predict']))
            min_q = min(min(cell_predictions['q_actual']), min(cell_predictions['q_predict']))

            regression_q = figure(width=400, height=400, title=f'Cell {cell}',
                                  x_axis_label='Actual Q (Ah)', y_axis_label='Predicted Q (Ah)',
                                  x_range=(min_q, max_q), y_range=(min_q, max_q))

            regression_q.circle(cell_predictions['q_actual'], cell_predictions['q_predict'], size=6, color='black',
                                fill_alpha=0.6)

            # Identity line
            regression_q.line([min_q, max_q], [min_q, max_q], line_width=2, color='green', legend_label='Identity Line')

            regression_q.legend.location = 'top_left'

            # Plot for dq
            p_dq = figure(title=f'Cell {cell} - RMSE:{cell_dq_rmse:.4f}, CRMSE(%):{cell_dq_crmse:.4f}',
                          x_axis_label='Cycle number',
                          y_axis_label='dQ (Ah)',
                          width=400,
                          height=400)

            p_dq.line('x', 'y_actual', source=source_dq, legend_label='Actual dq', line_width=2, line_color='green')
            p_dq.line('x', 'y_predict', source=source_dq, legend_label='Predicted dq', line_width=2,
                      line_color='orange')
            p_dq.circle('x', 'y_actual', source=source_dq, size=6, color='green', legend_label='Actual dq',
                        fill_alpha=0.6)
            p_dq.circle('x', 'y_predict', source=source_dq, size=6, color='orange', legend_label='Predicted dq',
                        fill_alpha=0.6)
            p_dq.legend.location = 'top_left'

            # Regression and Identity Line for dq_actual vs dq_predict
            max_dq = max(max(cell_predictions['dq_actual']), max(cell_predictions['dq_predict']))
            min_dq = min(min(cell_predictions['dq_actual']), min(cell_predictions['dq_predict']))

            regression_dq = figure(width=400, height=400, title=f'Cell {cell} - MAE:{cell_dq_mae:.4f}, MAPE(%):{cell_dq_mape:.4f}',
                                   x_axis_label='Actual dQ (Ah)', y_axis_label='Predicted dQ (Ah)',
                                   x_range=(min_dq, max_dq), y_range=(min_dq, max_dq))

            regression_dq.circle(cell_predictions['dq_actual'], cell_predictions['dq_predict'], size=6, color='black',
                                 fill_alpha=0.6)

            # Identity line
            regression_dq.line([min_dq, max_dq], [min_dq, max_dq], line_width=2, color='purple',
                               legend_label='Identity Line')

            regression_dq.legend.location = 'top_left'

            plots.append([p_q, regression_q, p_dq, regression_dq])

        grid = gridplot(plots)
        output_file(os.path.join(self.model_dir, output_filename))
        save(grid)

    def descale_value(self, data, min_max_values):
        """
        Scale the data using the global min/max values for each feature.

        :param data: The preprocessed data structured as a dictionary.
        :param min_max_values: The global min/max values for each feature.
        :return: The scaled data or the original data based on the configuration.
        """
        min_value = min_max_values['min']
        max_value = min_max_values['max']

        if max_value > min_value:
            return (data * (max_value - min_value)) + min_value
        else:
            return 0.0  # Handle edge case where min and max are the same
