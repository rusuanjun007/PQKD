from pathlib import Path

import torch
import torch_pruning as tp
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import qmobilenetv3
import qresnet
from dataset import create_dataloader, load_json

# set seed
torch.manual_seed(667)
torch.cuda.manual_seed(667)
torch.cuda.manual_seed_all(667)
torch.backends.cudnn.deterministic = True


def register_middle_outputs(model: nn.Module, name_dict: dict, type="student"):
    """
    Register the middle outputs of the 3x3 conv layer and adaptor layer.

    Args:
        model (nn.Module): The model to register the middle outputs
        name_dict (dict): The dictionary of the 3x3 conv layer and adaptor layer names
        type (str): The type of the model, student or teacher

    Returns:
        dict: The dictionary of the middle outputs
    """
    mid_out = {}

    def get_middle_output(name):
        def hook(model, input, output):
            mid_out[name] = output.detach()

        return hook

    # Register hooks for adaptor layers
    if type == "student":
        for name, info in name_dict.items():
            adaptor = dict(model.named_modules())[info[2]]
            adaptor.register_forward_hook(get_middle_output(name))
    else:
        for name, info in name_dict.items():
            x3_conv = dict(model.named_modules())[info[0]]
            x3_conv.register_forward_hook(get_middle_output(name))

    return mid_out


def kd_loss_fn() -> torch.Tensor:
    """
    Knowledge distillation loss function.

    Returns:
        torch.Tensor: The knowledge distillation loss
    """
    loss = 0
    for student_out, teacher_out in zip(
        student_mid_out.values(), teacher_mid_out.values()
    ):
        loss += nn.MSELoss()(student_out, teacher_out)
    return loss


def train(
    dataloader: torch.utils.data.DataLoader,
    student_model: nn.Module,
    teacher_model: nn.Module,
    loss_fn,
    kd_loss_fn,
    optimizer,
):
    """
    Train the student model with knowledge distillation.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for training
        student_model (nn.Module): The student model to train
        teacher_model (nn.Module): The teacher model
        loss_fn: The loss function
        kd_loss_fn: The knowledge distillation loss function
        optimizer: The optimizer
    """
    student_model.train()

    for batch, (X, y, idx) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        student_pred = student_model(X)
        with torch.no_grad():
            teacher_model(X)

        loss_reg = loss_fn(student_pred, y)
        loss_dk = config["kd_weight"] * kd_loss_fn()
        # print(loss_reg, loss_dk)

        loss = loss_reg + loss_dk

        # Back propagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            wandb.log({"train_batch_loss": loss})
            wandb.log({"train_batch_loss_reg": loss_reg})
            wandb.log({"train_batch_loss_dk": loss_dk})


def test(
    dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_fn, note=""
) -> torch.Tensor:
    """
    Test the model.

    Args:
        dataloader (torch.utils.data.DataLoader): The dataloader for testing
        model (nn.Module): The model to test
        loss_fn: The loss function
        note (str): The note for logging

    Returns:
        torch.Tensor: The average relative error
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

    return r_error * 100


if __name__ == "__main__":
    config = {
        "model": "qresnet50",  # qmobilenetv3_large, qresnet50
        "pruning_ratio": 0.9,  # 0.7, 0.9
        "iterative_steps": 10,
        "final_epoch": 100,
        "middle_epoch": 100,
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "kd_weight": 3e-4,  # 3e-2, 3e-4
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = torch.nn.HuberLoss()

    # Load the dataset
    data_root_path = Path("dataset")
    train_data = create_dataloader(
        load_json(data_root_path / "Aluminum_train.json")
        | load_json(data_root_path / "S355_Stainless_Steel_train.json"),
        data_length=256,
        batch_size=64,
        shuffle_flag=True,
    )
    aluminum_val_data = create_dataloader(
        load_json(data_root_path / "Aluminum_val.json"),
        data_length=256,
        batch_size=64,
        shuffle_flag=False,
    )
    s355_val_data = create_dataloader(
        load_json(data_root_path / "S355_Stainless_Steel_val.json"),
        data_length=256,
        batch_size=64,
        shuffle_flag=False,
    )

    # Load the model
    if config["model"] == "qresnet50":
        weight_path = "model_float/qresnet50.pth"
        student_model = qresnet.qresnet50(num_classes=1).to(device)
        teacher_model = qresnet.qresnet50(num_classes=1).to(device)
    elif config["model"] == "qmobilenetv3_large":
        weight_path = "model_float/qmobilenetv3_large.pth"
        student_model = qmobilenetv3.qmobilenet_v3_large(num_classes=1).to(device)
        teacher_model = qmobilenetv3.qmobilenet_v3_large(num_classes=1).to(device)

    # Load pre-trained weights
    student_model.load_state_dict(
        torch.load(weight_path, map_location=device),
        strict=True,
    )
    teacher_model.load_state_dict(
        torch.load(weight_path, map_location=device),
        strict=True,
    )

    # Set the teacher model frozen and no gradient
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Record the middle outputs of the 3x3 conv layer and adaptor layer
    new_to_original_names = {}
    for name, teacher_m in teacher_model.named_modules():
        if isinstance(teacher_m, torch.nn.Conv1d) and (
            teacher_m.kernel_size == (3,) or teacher_m.kernel_size == 3
        ):
            # New conv3 name: (Original conv3 name, Original out_channels, adaptor name)
            new_to_original_names[name + ".0"] = (
                name,
                teacher_m.out_channels,
                name + ".1",
            )

    # Collect the changes for replacing the 3x3 conv layer with sequential of 3x3 and adaptor layers
    changes = []
    for name, student_m in student_model.named_modules():
        if isinstance(student_m, torch.nn.Conv1d) and (
            student_m.kernel_size == (3,) or student_m.kernel_size == 3
        ):
            # Insert the kernel size of 1x1 conv layer
            channel_adaptor = torch.nn.Conv1d(
                student_m.out_channels,
                new_to_original_names[name + ".0"][1],
                kernel_size=(1,),
                stride=1,
                padding="same",
            )

            # replace the 3x3 conv layer with sequential of 3x3 and adaptor layers
            new_module = torch.nn.Sequential(student_m, channel_adaptor)
            changes.append((name, new_module))

    # Save name in changes as txt file
    adapter_names = [v[2] for v in new_to_original_names.values()]
    if not Path("pkd").exists():
        Path("pkd").mkdir(parents=True)
    with open(f"pkd/{config['model']}_adapter_names.txt", "w") as f:
        f.write(str(adapter_names))

    # Apply changes
    for name, new_module in changes:
        parent_module, child_name = name.rsplit(".", 1)
        parent_module = dict(student_model.named_modules())[parent_module]
        setattr(parent_module, child_name, new_module)
    student_model.to(device)

    # Set the ignored layers
    # DO NOT prune the final regression layer
    ignored_layers = []
    for student_m in student_model.modules():
        if isinstance(student_m, torch.nn.Linear) and student_m.out_features == 1:
            ignored_layers.append(student_m)

    # Do NOT prune the adaptor layer
    for name, info in new_to_original_names.items():
        ignored_layers.append(dict(student_model.named_modules())[info[2]])

    # Set pruner for the student model
    pruner = tp.pruner.MagnitudePruner(
        student_model,
        torch.randn(64, 1, 256).to(device),
        importance=tp.importance.MagnitudeImportance(p=2),
        iterative_steps=config["iterative_steps"],
        pruning_ratio=config[
            "pruning_ratio"
        ],  # remove 50% channels, ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
        ignored_layers=ignored_layers,
    )

    # Count the number of operations and parameters of the original model
    base_macs, base_nparams = tp.utils.count_ops_and_params(
        student_model, torch.randn(64, 1, 256).to(device)
    )

    run = wandb.init(
        project="PEC_PKD",
        name=config["model"],
        config=config,
        # mode="disabled",
    )

    # Test the original model
    aluminum_err = []
    s355_err = []
    params_percentages = []
    a_err = test(aluminum_val_data, teacher_model, loss_fn, note="Aluminum")
    s_err = test(s355_val_data, teacher_model, loss_fn, note="S355 Stainless Steel")

    aluminum_err.append(a_err.item())
    s355_err.append(s_err.item())
    params_percentages.append(100 - base_nparams / base_nparams * 100)

    # Prune the model iteratively and fine-tune with knowledge distillation
    for i in range(config["iterative_steps"]):
        pruner.step()

        macs, nparams = tp.utils.count_ops_and_params(
            student_model, torch.randn(64, 1, 256).to(device)
        )
        print(
            "Round %d/%d, Params: %.2f M"
            % (i + 1, config["iterative_steps"], nparams / 1e6)
        )

        # Fine-tuning with knowledge distillation
        student_mid_out = register_middle_outputs(
            student_model, new_to_original_names, "student"
        )
        teacher_mid_out = register_middle_outputs(
            teacher_model, new_to_original_names, "teacher"
        )

        optimizer = torch.optim.AdamW(
            student_model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        if i != config["iterative_steps"] - 1:
            epoch = config["middle_epoch"]
        else:
            epoch = config["final_epoch"]

        scheduler = CosineAnnealingLR(optimizer, T_max=epoch)

        for inner_i in tqdm(range(epoch)):
            train(
                train_data,
                student_model,
                teacher_model,
                loss_fn,
                kd_loss_fn,
                optimizer,
            )
            scheduler.step()
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

        # Test the fine-tuned model
        a_err = test(aluminum_val_data, student_model, loss_fn, note="Aluminum")
        s_err = test(s355_val_data, student_model, loss_fn, note="S355 Stainless Steel")

        aluminum_err.append(a_err.item())
        s355_err.append(s_err.item())
        params_percentages.append(100 - nparams / base_nparams * 100)

        # Save the model
        model_root = Path("pkd")
        # model.zero_grad()  # clear gradients to avoid a large file size
        # remove model middle ouptut hooks then save
        for name, module in student_model.named_modules():
            if isinstance(module, nn.Conv1d):
                module._forward_hooks.clear()

        torch.save(student_model, model_root / f"{config['model']}_{i}.pth")

    # Plot the result
    fig = plt.figure(figsize=(10, 5))
    plt.plot(params_percentages, aluminum_err, label="Aluminum")
    plt.plot(params_percentages, s355_err, label="S355 Stainless Steel")
    plt.legend()
    plt.xlabel("Pruning Ratio (%)")
    plt.ylabel("Average Relative Error (%)")
    fig.savefig(model_root / f"pkd_{config['model']}_result.png")

    # Save the list
    with open(model_root / f"pkd_{config['model']}_result.txt", "w") as f:
        f.write("Aluminum\n")
        f.write(str(aluminum_err) + "\n")
        f.write("S355 Stainless Steel\n")
        f.write(str(s355_err) + "\n")
        f.write("Params\n")
        f.write(str(params_percentages) + "\n")
