# Iberian Energy Price Prediction

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Deployment](#deployment)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Description
This project implements a Long Short-Term Memory (LSTM) neural network using PyTorch to predict energy prices in the Iberian market. It processes three days of hourly energy prices to predict the prices for the next 24 hours. The training data is sourced from the [Red Eléctrica de España (REE)](https://www.ree.es/es/apidatos) which provides an API service to retrieve Spanish energy market data.

## Installation
To set up the project environment, ensure you have Python 3.8+ installed, then follow these steps:
1. Clone the repository:
```
git clone https://github.com/RikiSot/LSTMEnergyPricePrediction.git
```
2. Navigate to the project directory:
```
cd iberian-energy-price-prediction
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage
To use the model for prediction, follow these steps:
1. Start the web interface using the Gradio library:
```
python app.py
```
2. Input the date in `YYYY-MM-DD` format to receive predictions for the next 24 hours of energy prices. Results will appear next to input data in a plot

You can also access the live Gradio app through the following link:

[Access the live app](https://huggingface.co/spaces/RikiSot/energy-price-predictor)

## Model Architecture
The LSTM model is constructed with configurable hyperparameters for the number of hidden units, layers, and dropout rate. The PyTorch framework facilitates the creation of the LSTM layers and the linear regression layer for output.
The model takes a time series input of 3 * 24 elements of hourly data scaled between 0-1 for all columns, so the shape of the input is [BatchSize, 3 * 24, N features (20)].

## Training
The model is trained using 2023 data from [REE](https://www.ree.es/es/apidatos). Data is previously preprocessed by:
- Adding categorical variables in one hot encoding from datetime info for each date. Features added are:
  - Day of the weekday (Monday-Sunday)
  - Day of the month
  - Month
- Create sequences of 3 * 24 elements for each feature of the data (value of the time series and categorical variables).

Training involves:
- Preprocessing and loading the data using custom DataLoader instances.
- Defining the LSTM model architecture with the specified hyperparameters.
- Using MSE as loss function and the Adam optimizer.
- Running the training loop for a set number of epochs, evaluating on the test set at each epoch.
- Process is supervised using TensorBoard for logging the loss and a plot of evaluation data vs predictions.
- An early stopping callback stops the training if validation loss stops improving after a certain number of epochs

## Results
With the hyperparameters that appear at hyperparameters.py model achieved a MSE loss of ~0.02 on the test set. Data is scaled between 0-1
so these are acceptable results!

It is crucial to acknowledge that the dataset utilized in this project corresponds to the temporal context of 2023.
As a result, the introduction of new data from the current year may yield variations from the training data, potentially leading to suboptimal performance in certain scenarios.
This aspect underscores the nature of this project as a portfolio endeavor, with the primary goal of refining my model development and deployment skills.

The versatility of the training pipeline is a notable feature, as it can be seamlessly adapted to accommodate diverse datasets.
This adaptability is facilitated by the ability to modify Dataset classes for each unique situation, ensuring the robustness and applicability of the model across different data contexts.

## Deployment
The model is deployed using the Gradio library, allowing for an interactive web interface where users can input a date to receive energy price predictions.
The deployment is hosted on Hugging Face spaces.

## License
This project is open-sourced under the MIT license.

## Acknowledgements
Special thanks to the Red Eléctrica de España for providing the energy market data API and to the open-source community for the tools and libraries used in this project.

# Future improvements
In a future update I would like to edit how the sequences are built, by only generating the sequence for the time series data, leaving the rest of categorical variables
as an scalar input to the model. In this case, model architecture would have been adapted to the input by
- One [Batch, Sequence-length, 1] LSTM input of the time series data
- One [Batch, Features] input to stack of Dense layers.