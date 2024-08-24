import os
import json
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bokeh.plotting import figure, output_file, save
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource

class ModelEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_dir = os.path.join(config['paths']['result'], config['experiment_name'])
        os.makedirs(self.model_dir, exist_ok=True)

    def evaluate_model(self, model, data: Dict[str, Any]):
        """
        Evaluate the model using the provided training and testing data.

        :param model: A trained model.
        :param data: A dictionary containing the features and target of training and testing cells.
        """
        aggregated_predictions = {'train': {'q_actual': [], 'q_predict': [], 'dq_actual': [], 'dq_predict': [], 'dates': []},
                                  'test': {'q_actual': [], 'q_predict': [], 'dq_actual': [], 'dq_predict': [], 'dates': []}}

        for cell_type in ['train_cells', 'test_cells']:
            cells_data = data[cell_type]
            predictions = {}

            for cell, cell_data in cells_data.items():
                # Evaluate the model for this single cell
                predictions[cell], cell_metrics = self._evaluate_single_cell(model, cell_data)

                # Accumulate all actual, predicted values, dq values, and dates
                aggregated_predictions[cell_type.split('_')[0]]['q_actual'].extend(predictions[cell]['q_actual'])
                aggregated_predictions[cell_type.split('_')[0]]['q_predict'].extend(predictions[cell]['q_predict'])
                aggregated_predictions[cell_type.split('_')[0]]['dq_actual'].extend(predictions[cell]['dq_actual'])
                aggregated_predictions[cell_type.split('_')[0]]['dq_predict'].extend(predictions[cell]['dq_predict'])
                aggregated_predictions[cell_type.split('_')[0]]['dates'].extend(predictions[cell]['dates'])

                # Save metrics for the individual cell
                self._save_metrics(predictions[cell], f'{cell_type}_{cell}_predictions.json')

            # Calculate aggregated metrics over all cells for this cell type (train/test)
            metrics = self._calculate_aggregated_metrics(aggregated_predictions[cell_type.split('_')[0]])

            # Save the overall metrics for the cell type
            with open(os.path.join(self.model_dir, f'{cell_type}_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=4)

            # Plot the results for this cell type
            self._plot_results(predictions, f'{cell_type}_results.html')


    def _evaluate_single_cell(self, model, cell_data: Dict[str, Any]):
        """
        Evaluate a single cell's performance.

        :param model: A trained model.
        :param cell_data: A dictionary containing the features, target, and ground truth cycles.
        :return predictions: A dictionary containing predictions, actual values, and dates.
        :return cell_metrics: A dictionary containing the error metrics for this cell.
        """
        predictions = {}

        features = cell_data['features']
        dq_actual = np.array(cell_data['target'])  # Actual dq values
        dq_predict = model.predict(features).flatten()  # Predicted dq values

        ground_truth = cell_data['ground_truth_cycle']
        nominal_capacity = self.config['nominal_capacity']

        q_actual = np.array([cycle['capacity'] for cycle in ground_truth])
        dates = [cycle['date'] for cycle in ground_truth]
        q_prev_step = [nominal_capacity] + q_actual[:-1].tolist()
        q_predict = [q_prev + dq_current_predict for q_prev, dq_current_predict in zip(q_prev_step, dq_predict.tolist())]

        # Store predictions, actual values, dq values, and dates
        predictions['q_actual'] = q_actual.tolist()
        predictions['q_predict'] = q_predict
        predictions['dq_actual'] = dq_actual.tolist()
        predictions['dq_predict'] = dq_predict.tolist()
        predictions['dates'] = [date.isoformat() for date in dates]  # Convert datetime to string for JSON serialization

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
                'mae': mean_absolute_error(dq_actual, dq_predict),
                'mape': np.mean(np.abs((dq_actual - dq_predict) / dq_actual)) * 100,
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
                'mae': mean_absolute_error(dq_actual, dq_predict),
                'mape': np.mean(np.abs((dq_actual - dq_predict) / dq_actual)) * 100,
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

    def _plot_results(self, predictions: Dict[str, Any], output_filename: str):
        """
        Plot the predictions versus actual values using Bokeh and save the results.

        :param predictions: The predictions dictionary containing actual and predicted values along with dates.
        :param output_filename: The filename to save the HTML plot.
        """
        plots = []
        for cell, cell_predictions in predictions.items():
            source_q = ColumnDataSource(data=dict(
                x=[np.datetime64(date) for date in cell_predictions['dates']],
                y_actual=cell_predictions['q_actual'],
                y_predict=cell_predictions['q_predict']
            ))

            source_dq = ColumnDataSource(data=dict(
                x=[np.datetime64(date) for date in cell_predictions['dates']],
                y_actual=cell_predictions['dq_actual'],
                y_predict=cell_predictions['dq_predict']
            ))

            # Plot for capacity (q)
            p_q = figure(title=f'Cell {cell} - Predicted vs. Actual Capacity (Q)',
                         x_axis_label='Date',
                         y_axis_label='Capacity (Ah)',
                         x_axis_type='datetime',
                         width=400,
                         height=400)

            p_q.line('x', 'y_actual', source=source_q, legend_label='Actual Q', line_width=2, line_color='blue')
            p_q.line('x', 'y_predict', source=source_q, legend_label='Predicted Q', line_width=2, line_color='red')
            p_q.circle('x', 'y_actual', source=source_q, size=6, color='blue', legend_label='Actual Q', fill_alpha=0.6)
            p_q.circle('x', 'y_predict', source=source_q, size=6, color='red', legend_label='Predicted Q', fill_alpha=0.6)
            p_q.legend.location = 'top_left'

            # Plot for dq
            p_dq = figure(title=f'Cell {cell} - Predicted vs. Actual dq',
                          x_axis_label='Date',
                          y_axis_label='dq (Ah)',
                          x_axis_type='datetime',
                          width=400,
                          height=400)

            p_dq.line('x', 'y_actual', source=source_dq, legend_label='Actual dq', line_width=2, line_color='green')
            p_dq.line('x', 'y_predict', source=source_dq, legend_label='Predicted dq', line_width=2, line_color='orange')
            p_dq.circle('x', 'y_actual', source=source_dq, size=6, color='green', legend_label='Actual dq', fill_alpha=0.6)
            p_dq.circle('x', 'y_predict', source=source_dq, size=6, color='orange', legend_label='Predicted dq', fill_alpha=0.6)
            p_dq.legend.location = 'top_left'

            plots.append([p_q, p_dq])

        grid = gridplot(plots)
        output_file(os.path.join(self.model_dir, output_filename))
        save(grid)
