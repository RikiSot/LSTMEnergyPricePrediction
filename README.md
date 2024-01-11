# Iberian Energy Price Prediction

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Description
This project implements a Long Short-Term Memory (LSTM) neural network using PyTorch to predict energy prices in the Iberian market. It processes three days of hourly energy prices to predict the prices for the next 24 hours. The training data is sourced from the [Red Eléctrica de España (REE)](https://www.ree.es/es/apidatos) which provides an API service to retrieve Spanish energy market data.

## Installation
To set up the project environment, ensure you have Python 3.8+ installed, then follow these steps:
1. Clone the repository:
```
git clone https://github.com/your-username/iberian-energy-price-prediction.git
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

[Access the live app](https://your-gradio-app-link)

## Model Architecture
The LSTM model is constructed with configurable hyperparameters for the number of hidden units, layers, and dropout rate. The PyTorch framework facilitates the creation of the LSTM layers and the linear regression layer for output.

## Training
The model is trained using the data from [REE](https://www.ree.es/es/apidatos), with the training process supervised using TensorBoard for logging the results. Training involves:
- Preprocessing and loading the data using custom DataLoader instances.
- Defining the LSTM model architecture with the specified hyperparameters.
- Using an MSE loss function and the Adam optimizer.
- Running the training loop for a set number of epochs, evaluating on the test set at each epoch.

## Deployment
The model is deployed using the Gradio library, allowing for an interactive web interface where users can input a date to receive energy price predictions. The deployment is hosted on Hugging Face spaces.

## License
This project is open-sourced under the MIT license.

## Acknowledgements
Special thanks to the Red Eléctrica de España for providing the energy market data API and to the open-source community for the tools and libraries used in this project.
