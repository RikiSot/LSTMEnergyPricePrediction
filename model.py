from torch import nn


class LSTMModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_units=50, num_layers: int = 1, dropout=0) -> None:
        super().__init__()
        self.n_hidden = hidden_units
        self.scaler = None  # Scaler used during training
        # Build layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.regressor = nn.Linear(hidden_units, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        out, (hidden, _) = self.lstm(x)
        predictions = self.regressor(out)
        return predictions[:, -1, :]
