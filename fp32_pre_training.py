from pathlib import Path

import torch
import wandb
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import qmobilenetv3
import qresnet
from dataset import create_dataloader, load_json
from model_baseline import CNN1D, LSTM, MLP

# set seed
torch.manual_seed(667)
torch.cuda.manual_seed(667)
torch.cuda.manual_seed_all(667)
torch.backends.cudnn.deterministic = True


def train(
    dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_fn, optimizer
) -> None:
    """
    Train the model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for training
        model (nn.Module): The model to train
        loss_fn: The loss function
        optimizer: The optimizer
    """
    model.train()
    for batch, (X, y, idx) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            wandb.log({"train_batch_loss": loss})


def test(dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_fn, note=""):
    """
    Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for testing
        model (nn.Module): The model to test
        loss_fn: The loss function
        note (str): The note for logging
    """
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0
    r_error = 0
    with torch.no_grad():
        for X, y, idx in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)

            # Exclude y_true of zero values
            mask = y != 0
            r_error += torch.abs((y - pred) / y)[mask].mean()

    test_loss /= num_batches
    r_error /= num_batches

    wandb.log({f"{note}_val_ave_loss": test_loss})
    wandb.log({f"{note}_val_relative_error": r_error})


if __name__ == "__main__":
    device = (
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )

    config = {
        "epochs": 100,
        "model": "qresnet50",  # qresnet18, qmobilenetv3_small, qresnext50_32x4d
        "data_length": 256,
        "batch_size": 64,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "loss": "HuberLoss",  # HuberLoss
        "test_epoch": 10,
    }

    config["device"] = device

    # Load data
    data_root_path = Path("dataset")

    train_data = create_dataloader(
        load_json(data_root_path / "Aluminum_train.json")
        | load_json(data_root_path / "S355_Stainless_Steel_train.json"),
        data_length=config["data_length"],
        batch_size=config["batch_size"],
        shuffle_flag=True,
    )

    aluminum_val_data = create_dataloader(
        load_json(data_root_path / "Aluminum_val.json"),
        data_length=config["data_length"],
        batch_size=64,
        shuffle_flag=False,
    )
    s355_val_data = create_dataloader(
        load_json(data_root_path / "S355_Stainless_Steel_val.json"),
        data_length=config["data_length"],
        batch_size=64,
        shuffle_flag=False,
    )

    # Load model
    if "mobilenet" in config["model"]:
        model = qmobilenetv3.qmobilenet_v3_large(num_classes=1).to(device)
    elif "resnet" in config["model"]:
        model = qresnet.qresnet50(num_classes=1).to(device)
    elif "resnext" in config["model"]:
        model = qresnet.qresnext50_32x4d(num_classes=1).to(device)
    elif config["model"] == "MLP":
        model = MLP(256, 1840).to(device)
    elif config["model"] == "CNN1D":
        model = CNN1D(1, 1130).to(device)
    elif config["model"] == "LSTM":
        model = LSTM(1, 440).to(device)

    print(model)

    if config["loss"] == "L2":
        loss_fn = nn.MSELoss()
    elif config["loss"] == "HuberLoss":
        loss_fn = nn.HuberLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    run = wandb.init(
        project="PEC_Float",
        name=config["model"],
        config=config,
        # mode="disabled",
    )
    for t in range(1, config["epochs"] + 1):
        print(f"Epoch {t}")
        train(train_data, model, loss_fn, optimizer)

        test_model = model

        if t % config["test_epoch"] == 0 or t == config["epochs"]:
            test(aluminum_val_data, test_model, loss_fn, "Aluminum")
            test(s355_val_data, test_model, loss_fn, "S355_Stainless_Steel")

        scheduler.step()
        wandb.log({"lr": optimizer.param_groups[0]["lr"]})

    # Save model
    model_root = Path("model_float")
    if not model_root.exists():
        model_root.mkdir(parents=True)

    torch.save(model.state_dict(), model_root / f"{config['model']}.pth")
