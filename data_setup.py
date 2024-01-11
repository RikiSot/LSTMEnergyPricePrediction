from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class DemandDataset(Dataset):

    def __init__(self, data: pd.DataFrame, output_size: int = 24, target_column='value', num_sequences: int = 24, scaler: MinMaxScaler = None):
        if scaler is None:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = scaler
        data = self.preprocess_data(data, self.scaler)
        self.ts = data.index[num_sequences:-output_size]
        self.sequences = self.create_sequences(data, target_column,
                                               sequence_length=num_sequences, num_predictions=output_size)
        self.features = len(data.columns)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        sequence, labels = self.sequences[index]
        return torch.tensor(sequence.to_numpy()).type(torch.float), torch.tensor(labels.to_numpy()).type(torch.float)

    @staticmethod
    def preprocess_data(data: pd.DataFrame, scaler, target_col='value'):
        # Drop NaN values (replace by ffill)
        data[target_col] = data[target_col].ffill()
        # Scale the "value" column between 0 and 1
        data[target_col] = scaler.fit_transform(data[[target_col]])

        # Apply one-hot encoding to categorical columns ("day" and "month")
        categorical_columns = ["day", "month"]
        data['day'] = data.index.weekday
        data['month'] = data.index.month
        data = data[['value', 'day', 'month']]

        encoder = OneHotEncoder(sparse_output=False, drop='first', dtype=int)  # Use 'drop' to avoid multicollinearity
        encoded_data = encoder.fit_transform(data[categorical_columns])

        # Manually create column names for one-hot encoded columns
        encoded_columns = []
        for i, col in enumerate(categorical_columns):
            unique_values = sorted(data[col].unique())
            unique_values.remove(data[col].mode().values[0])
            for value in unique_values:
                encoded_columns.append(f"{col}_{value}")

        # Create a DataFrame with the one-hot encoded categorical data
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns, index=data.index)

        # Create columns for all days and months
        for i in range(7):
            if f"day_{i}" not in encoded_df.columns:
                encoded_df[f"day_{i}"] = 0
        for i in range(1, 13):
            if f"month_{i}" not in encoded_df.columns:
                encoded_df[f"month_{i}"] = 0

        # Concatenate the scaled "value" column and the one-hot encoded categorical data
        data = pd.concat([data[["value"]], encoded_df], axis=1)
        # Reorder the columns
        cols = ['value'] + [f"day_{i}" for i in range(7)] + [f"month_{i}" for i in range(1, 13)]
        data = data.reindex(cols, axis=1)
        return data

    @staticmethod
    def create_sequences(input_data: pd.DataFrame, target_column, sequence_length, num_predictions):
        sequences = []
        data_size = len(input_data)

        for i in tqdm(range(data_size - sequence_length - num_predictions)):
            sequence = input_data[i:i + sequence_length]
            labels = input_data.iloc[i + sequence_length:i + sequence_length + num_predictions][target_column]
            sequences.append((sequence, labels))

        return sequences


def create_datasets(data_dir: str, num_sequences: int, output_size: int) -> (DemandDataset, DemandDataset, DemandDataset):
    """Creates training, testing and evaluation DataSets.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets.

    Args:
        data_dir: Path to data directory.
        num_sequences: number of elements in the moving window
        output_size: number of element to predict

    Returns:
    A tuple of (train_dataset, test_dataset).

    Example usage:
      train_dataset, test_dataset = \
        = create_dataloaders(data_dir=path_to_file,
                             num_sequences=32)
    """

    # Load data
    data = pd.read_csv(data_dir, index_col='timeStamp', parse_dates=['timeStamp'])
    data.index = pd.to_datetime(data.index, utc=True)
    data.index = data.index.tz_convert('Europe/Madrid')
    train, test = train_test_split(data, shuffle=False, test_size=0.2)
    test, eval = train_test_split(test, shuffle=False, test_size=0.5)

    train_dataset = DemandDataset(data=train, num_sequences=num_sequences, output_size=output_size)
    # Get scaler from train data
    scaler = train_dataset.scaler
    test_dataset = DemandDataset(data=test, num_sequences=num_sequences, output_size=output_size, scaler=scaler)
    eval_dataset = DemandDataset(data=eval, num_sequences=num_sequences, output_size=output_size, scaler=scaler)
    return train_dataset, test_dataset, eval_dataset


def create_dataloaders(
        data_dir: str,
        batch_size: int,
        num_sequences: int,
        output_size: int,
        num_workers: int = 0) -> (DataLoader, DataLoader, DataLoader):
    """Creates training, testing and evaluation DataLoaders.

    Takes in a training directory and testing directory path and turns
    them into PyTorch Datasets and then into PyTorch DataLoaders.

    Args:
    data: Path to data directory.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_sequences: number of elements in the moving window
    output_size: number of elements to predict in advance
    num_workers: An integer for number of workers per DataLoader.

    Returns:
    A tuple of (train_dataloader, test_dataloader)
    Example usage:
      train_dataloader, test_dataloader, eval_dataloader = \
        = create_dataloaders(data_dir=path_to_file,
                             batch_size=32,
                             num_sequences=24,
                             num_workers=4)
    """
    train_dataset, test_dataset, eval_dataset = create_datasets(data_dir, num_sequences, output_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                 pin_memory=True)

    return train_dataloader, test_dataloader, eval_dataloader
