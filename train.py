"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import torch

import data_setup
import engine
import model
import utils

# Setup hyperparameters
from hyperparameters import *

# Setup directories
data_dir = "data/price_data.csv"

# name of the experiment by hyperparameters
exp_name = f'LSTM_{NUM_LAYERS}_layers_{HIDDEN_UNITS}_units_{NUM_EPOCHS}_epochs'

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Tensorboard session
writer = utils.create_writer(experiment_name='price_predictor', model_name=exp_name)

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, eval_dataloader = data_setup.create_dataloaders(data_dir,
                                                                                   batch_size=BATCH_SIZE,
                                                                                   num_sequences=NUM_SEQUENCES,
                                                                                   output_size=OUTPUT_SIZE)
# Get input size of the net
INPUT_SIZE = train_dataloader.dataset.features

# Create model with help from model.py
model = model.LSTMModel(
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    num_layers=NUM_LAYERS,
    hidden_units=HIDDEN_UNITS,
    dropout=DROPOUT,
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device,
             writer=writer,
             patience=10)

engine.validation(model=model,
                  eval_dataloader=eval_dataloader,
                  device=device,
                  writer=writer)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name=f'{exp_name}.pth')

writer.close()
