"""
Utility functions to make predictions.

Main reference for code creation: https://www.learnpytorch.io/06_pytorch_transfer_learning/#6-make-predictions-on-images-from-the-test-set
"""
from datetime import datetime, timedelta
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from download_price_data import get_price_data

from data_setup import DemandDataset
from model import LSTMModel

from hyperparameters import *
import pickle

from utils import plot_prediction

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_scaler(path: str):
    """Loads a MinMaxScaler from a file."""
    with open(path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def predict(date_str: str) -> (plt.Figure, float):
    """
    Returns the prediction of the energy price of the next day for a given date

    Args:
        date_str: date to use as input of the model
    """
    # Start the timer
    date = pd.to_datetime(date_str)
    start_time = timer()

    # Create model file path
    model_dir = Path('./models/model')
    model_file_path = model_dir / 'model.pth'

    # Create scaler file path
    scaler_file_path = model_dir / 'scaler.pkl'

    # Get the data for the day provided
    end_date = datetime(date.year, date.month, date.day, date.hour)
    start_date = end_date - timedelta(hours=NUM_SEQUENCES)

    if end_date >= datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
        print('Prediction requested is for new data. No baseline will be plotted')
        data = get_price_data(start_date, end_date)
        extended_data = []
    else:
        print('Prediction requested can be compared to actual data.')
        extended_end_date = end_date + timedelta(hours=OUTPUT_SIZE)
        data = get_price_data(start_date, extended_end_date)
        extended_data = data[-OUTPUT_SIZE:].to_numpy().reshape(-1, 1)
        data = data[:-OUTPUT_SIZE]

    scaler = load_scaler(scaler_file_path)
    X, index = prepare_prediction_data(data, NUM_SEQUENCES, scaler, target_col='value')
    # Take only last prediction if len of data is greater than model
    INPUT_SIZE = X.shape[-1]

    # Put model into evaluation mode and turn on inference mode
    model = LSTMModel(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        num_layers=NUM_LAYERS,
        hidden_units=HIDDEN_UNITS,
        dropout=DROPOUT,
    )

    model.load_state_dict(torch.load(model_file_path))
    model = model.to(device)
    model.eval()
    X = X.to(device)

    with torch.inference_mode():
        yhat = model(X).squeeze()

    yhat = yhat.cpu().numpy()
    yhat = np.expand_dims(yhat, axis=-1)
    X = X.cpu().numpy()[0, :, :][:, 0]
    X = np.expand_dims(X, axis=-1)
    # Invert the scaler
    yhat = scaler.inverse_transform(yhat)
    X = scaler.inverse_transform(X)
    # Plot results of the
    # Calculate the prediction time
    pred_time = round(timer() - start_time, 5)
    fig = plot_prediction(yhat, X, index, extended_data)
    return fig, pred_time


def prepare_prediction_data(data: pd.DataFrame, num_sequences: int, scaler: MinMaxScaler, target_col='value') -> (torch.tensor, list):
    # Preprocess data as it was done during training
    X = DemandDataset.preprocess_data(data, scaler, target_col=target_col)
    X = X[-num_sequences:]
    X = torch.tensor(X.to_numpy()).type(torch.float).unsqueeze(0)
    index = data.index[-num_sequences:]
    return X, index


if __name__ == '__main__':
    date = datetime.now() - timedelta(days=30)
    date = date.strftime('%Y-%m-%d')
    predict(date)
