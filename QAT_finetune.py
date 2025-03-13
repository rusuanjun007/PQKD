import copy
from pathlib import Path

import torch
import wandb
from torch import nn
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from torch.optim.lr_scheduler import CosineAnnealingLR

import qmobilenetv3
import qresnet
from dataset import create_dataloader, load_json

# set seed
torch.manual_seed(667)
torch.cuda.manual_seed(667)
torch.cuda.manual_seed_all(667)
torch.backends.cudnn.deterministic = True


def train(
    dataloader: torch.utils.data.DataLoader,
    model: torch.fx.GraphModule,
    loss_fn,
    optimizer,
):
    """
    Train the quantized model for one epoch.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for training
        model (torch.fx.GraphModule): The quantized model to train
        loss_fn: The loss function
        optimizer: The optimizer
    """
    torch.ao.quantization.move_exported_model_to_train(model)
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


def test(
    dataloader: torch.utils.data.DataLoader,
    model: torch.fx.GraphModule,
    loss_fn,
    note="",
):
    """
    Test the quantized model.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for testing
        model (torch.fx.GraphModule): The quantized model to test
        loss_fn: The loss function
        note (str): The note for logging
    """
    num_batches = len(dataloader)
    torch.ao.quantization.move_exported_model_to_eval(model)

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

    print(f"{note} test_loss: {test_loss}, r_error: {r_error}")
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
        "model": "qresnet50_pruned",  # qresnet50_pruned, qmobilenetv3_large_pruned
        "device": device,
        "data_length": 256,
        "batch_size": 64,
        "lr": 1e-2,
        "weight_decay": 0,
        "loss": "HuberLoss",  # HuberLoss, L2
    }
    config["test_epoch"] = config["epochs"] // 10 if config["epochs"] > 10 else 1

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
    if "qresnet50_pruned" == config["model"]:
        model_path = "pkd/qresnet50_0.pth"
        float_model = torch.load(model_path, weights_only=False)
    elif "qmobilenetv3_large_pruned" == config["model"]:
        model_path = "pkd/qmobilenetv3_large_0.pth"
        float_model = torch.load(model_path, weights_only=False)
    else:
        if "qmobilenet" in config["model"]:
            float_model = qmobilenetv3.qmobilenet_v3_large(num_classes=1).to(device)
        elif "qresnet" in config["model"]:
            float_model = qresnet.qresnet50(num_classes=1).to(device)
        elif "qresnext" in config["model"]:
            float_model = qresnet.qresnext50_32x4d(num_classes=1).to(device)

        # Load the parameters of the float model
        weight_path = f"model_float/{config['model']}.pth"
        float_model.load_state_dict(
            torch.load(weight_path, map_location=device), strict=True
        )
        print(f"Load float model from {weight_path}")

    # Step 1. program capture. This is available for pytorch 2.5+
    exported_model = torch.export.export_for_training(
        float_model, (torch.randn(64, 1, 256).to(device),)
    ).module()

    # Step 2. quantization-aware training
    # backend developer will write their own Quantizer and expose methods to allow
    # users to express how they want the model to be quantized
    qconfig = get_symmetric_quantization_config(is_qat=True, is_per_channel=True)
    print(f"qconfig.weight: {qconfig.weight}")
    print(f"qconfig.input_activation: {qconfig.input_activation}")
    print(f"qconfig.output_activation: {qconfig.output_activation}")
    quantizer = XNNPACKQuantizer().set_global(qconfig)

    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    # print(prepared_model)

    if config["loss"] == "L2":
        loss_fn = nn.MSELoss()
    elif config["loss"] == "HuberLoss":
        loss_fn = nn.HuberLoss()

    optimizer = torch.optim.AdamW(
        prepared_model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])

    run = wandb.init(
        project="PEC_QAT_Finetune",
        name=config["model"],
        config=config,
        # mode="disabled",
    )

    # Test the quantized model before training
    prepared_model_copy = copy.deepcopy(prepared_model)
    quantized_model = convert_pt2e(prepared_model_copy)
    test(aluminum_val_data, quantized_model, loss_fn, "Aluminum")
    test(s355_val_data, quantized_model, loss_fn, "S355_Stainless_Steel")

    for t in range(1, config["epochs"] + 1):
        print(f"Epoch {t}")
        train(train_data, prepared_model, loss_fn, optimizer)

        # Test the quantized model
        # if t % config["test_epoch"] == 0 or t == config["epochs"]:
        if t == config["epochs"]:
            prepared_model_copy = copy.deepcopy(prepared_model)
            quantized_model = convert_pt2e(prepared_model_copy)
            test(aluminum_val_data, quantized_model, loss_fn, "Aluminum")
            test(s355_val_data, quantized_model, loss_fn, "S355_Stainless_Steel")

        scheduler.step()
        wandb.log({"lr": optimizer.param_groups[0]["lr"]})

    # Save prepared model
    model_root = Path("qat")
    if not model_root.exists():
        model_root.mkdir(parents=True)

    if "pruned" not in config["model"]:
        # Only save the weights of the model
        torch.save(
            prepared_model.state_dict(), model_root / f"{config['model']}_qat.pth"
        )
    else:
        pt2e_quantized_model_file_path = model_root / f"{config['model']}_qat.pth"
        # Export the model
        quantized_model = convert_pt2e(prepared_model)

        # capture the model to get an ExportedProgram
        quantized_ep = torch.export.export(
            quantized_model, (torch.randn(64, 1, 256).to(device),)
        )
        # save an ExportedProgram
        torch.export.save(quantized_ep, pt2e_quantized_model_file_path)
