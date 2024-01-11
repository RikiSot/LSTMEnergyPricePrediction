"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset
from utils import plot_validation


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device
               ) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
      model: A PyTorch model to be trained.
      dataloader: A DataLoader instance for the model to be trained on.
      loss_fn: A PyTorch loss function to minimize.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of training loss and training accuracy metrics.
      In the form (train_loss, train_accuracy). For example:

      (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches (if classifier)
        if isinstance(loss_fn, CrossEntropyLoss):
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        else:
            train_acc = None

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    if train_acc is not None:
        train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
      model: A PyTorch model to be tested.
      dataloader: A DataLoader instance for the model to be tested on.
      loss_fn: A PyTorch loss function to calculate loss on the test data.
      device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
      A tuple of testing loss and testing accuracy metrics.
      In the form (test_loss, test_accuracy). For example:

      (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            if isinstance(loss_fn, CrossEntropyLoss):
                test_pred_labels = test_pred.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
            else:
                test_acc = None

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    if test_acc is not None:
        test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device | str,
          writer: SummaryWriter = None) -> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
      model: A PyTorch model to be trained and tested.
      train_dataloader: A DataLoader instance for the model to be trained on.
      test_dataloader: A DataLoader instance for the model to be tested on.
      optimizer: A PyTorch optimizer to help minimize the loss function.
      loss_fn: A PyTorch loss function to calculate loss on both datasets.
      epochs: An integer indicating how many epochs to train for.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
      A dictionary of training and testing loss as well as training and
      testing accuracy metrics. Each metric has a value in a list for
      each epoch.
      In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]}
      For example if training for epochs=2:
                   {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]}
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # Loop through training and testing steps for a number of epochs
    model.scaler = train_dataloader.dataset.scaler
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

        # Print out what's happening every 10 epochs
        if (epoch + 1) % 10 == 0:
            if test_acc is not None and train_acc is not None:
                print(
                    f"Epoch: {epoch + 1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                )
            else:
                print(
                    f"Epoch: {epoch + 1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                )

        # Add loss results to SummaryWriter
        if writer is not None:
            writer.add_scalars(main_tag="Loss",
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)

            # Add accuracy results to SummaryWriter
            if test_acc is not None:
                writer.add_scalars(main_tag="Accuracy",
                                   tag_scalar_dict={"train_acc": train_acc,
                                                    "test_acc": test_acc},
                                   global_step=epoch)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results


def validation(model: torch.nn.Module,
               eval_dataloader: torch.utils.data.DataLoader,
               device: torch.device | str,
               writer: SummaryWriter,
               ) -> (torch.tensor, torch.tensor):
    """
    Takes a dataloader with inputs and labels and compare all predictions vs true values

    Args:
      model: A PyTorch model to be trained and tested.
      eval_dataloader: A DataLoader instance for the model to be evaluated on.
      device: A target device to compute on (e.g. "cuda" or "cpu").
      writer: A SummaryWriter() instance to log model results to.

    Returns:
        tuple of tensors of predictions, values
    """
    model = model.to(device)
    with torch.inference_mode():
        predictions = torch.tensor([]).to(device)
        values = torch.tensor([]).to(device)
        for batch, (X, y) in enumerate(eval_dataloader):
            X, y = X.to(device), y.to(device)
            model.eval()
            yhat = model(X)
            # Append predictions and values as tensors
            predictions = torch.cat([predictions, yhat], dim=0)
            values = torch.cat([values, y], dim=0)
    try:
        index = eval_dataloader.dataset.ts
    except:
        index = None
    predictions = predictions.cpu().numpy()
    values = values.cpu().numpy()
    fig = plot_validation(predictions, values, index)
    writer.add_figure('Predictions vs actuals', fig)
    return predictions, values
