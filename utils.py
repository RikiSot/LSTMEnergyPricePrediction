"""
Contains various utility functions for PyTorch model training and saving.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

from matplotlib.figure import Figure
from torch.utils.tensorboard.writer import SummaryWriter
import matplotlib.pyplot as plt
import pickle


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
    """
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    root, _ = os.path.splitext(model_name)  # Remove the extension
    model_folder_path = target_dir_path / root
    model_folder_path.mkdir(parents=True, exist_ok=True)

    # Save the model state_dict()
    model_save_path = model_folder_path / model_name
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    # Also save MinMaxScaler used during training
    scaler_save_path = model_folder_path / 'scaler.pkl'
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(model.scaler, f)


def plot_validation(predictions, values, index=None) -> Figure:
    """
    Plot a graph of real values vs predictions

    Args:
        predictions: list of predictions
        values: list of real values
        index: index of the data. If None, it will use range(len(data))

    """
    if index is None:
        index = range(len(predictions))
    # If predictions is a multistep prediction, we will only plot the first one
    if predictions.shape[-1] > 1:
        predictions = predictions[:, 0]
        values = values[:, 0]
    # Plotting
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot data
    ax.plot(index, values, label=f'True Values', linestyle='-', color='blue', linewidth=2.0)
    ax.plot(index, predictions, label=f'Predictions', linestyle='-.', color='red', alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('Time Series Predictions vs True Labels')
    ax.legend()
    return fig


def plot_prediction_old(yhat, X, index) -> Figure:
    """
    Plot a graph of real values vs predictions

    Args:
        yhat: list of predictions
        X: list of real values
        index: index of the data

    """
    # Plotting
    fig, ax = plt.subplots(figsize=(18, 12))

    # Plot data
    ax.plot(index, X, label=f'True Values', linestyle='-', color='blue', linewidth=2.0)
    # Number of hours to add
    num_hours = len(yhat)

    # Create a new DatetimeIndex for the predictions
    yhat_index = pd.date_range(start=index[-1], periods=num_hours + 1, freq='H')
    predictions = np.concatenate((X[-1:], yhat))

    ax.plot(yhat_index, predictions, label=f'Predictions', linestyle='-.', color='red', alpha=0.8, linewidth=1.5)

    ax.set_xlabel('Time')
    ax.set_ylabel('Values')
    ax.set_title('Time Series Predictions vs True Labels')
    ax.legend()
    plt.show()
    return fig


def plot_prediction(yhat, X, index, prediction_values: list[float] | None = None) -> go.Figure:
    """
    Plot a graph of real values vs predictions

    Args:
        yhat: list of predictions
        X: list of real values
        index: index of the data
        prediction_values: Optional. Actual values of the prediction

    """

    # Number of hours to add
    num_hours = len(yhat)

    # Create a new DatetimeIndex for the predictions
    yhat_index = pd.date_range(start=index[-1], periods=num_hours + 1, freq='H')
    predictions = np.concatenate((X[-1:], yhat))

    # Create a trace for the true values
    trace0 = go.Scatter(
        x=index,
        y=X.squeeze(),
        mode='lines',
        name='True Values',
        line=dict(color='rgb(0,100,80)')
    )

    # Create a trace for the predictions
    trace1 = go.Scatter(
        x=yhat_index,
        y=predictions.squeeze(),
        mode='lines',
        name='Predictions',
        line=dict(color='rgb(205,12,24)', width=4)
    )

    traces = [trace0, trace1]

    # Add actual values if needed
    if prediction_values is not None:
        prediction_values = np.concatenate((X[-1:], prediction_values))
        trace2 = go.Scatter(
            x=yhat_index,
            y=prediction_values.squeeze(),
            mode='lines',
            name='Real values',
            line=dict(color='rgb(0,0,255)', width=4)
        )
        traces.append(trace2)

    # Define the layout
    layout = go.Layout(
        title='Energy prices prediction',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Eur/MWh'),
        template='seaborn'
    )

    # Create a figure and add traces
    fig = go.Figure(data=traces, layout=layout)
    fig.show()
    return fig


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str = None) -> SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")  # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_model = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), self.path)
        self.best_model = model.state_dict()
        self.val_loss_min = val_loss
