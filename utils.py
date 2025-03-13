from pathlib import Path

import torch

from dataset import create_dataloader, load_json


def calibrate(model: torch.fx.GraphModule, device: str):
    """
    Calibrate the model with the training dataset.

    Args:
        model: The model to be calibrated.
        device: The device to run the model.
    """
    train_dataloader = create_dataloader(
        load_json(Path("dataset") / "Aluminum_train.json")
        | load_json(Path("dataset") / "S355_Stainless_Steel_train.json"),
        data_length=256,
        batch_size=64,
        shuffle_flag=True,
    )

    torch.ao.quantization.move_exported_model_to_eval(model)
    with torch.no_grad():
        for X, y, idx in train_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)


def test(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module | torch.fx.GraphModule,
    loss_fn,
    device: str,
    note="",
    is_graph: bool = False,
    verbose: bool = True,
):
    """
    Test the model with the testing dataset.

    Args:
        dataloader: The testing dataset.
        model: The model to be tested.
        loss_fn: The loss function.
        device: The device to run the model.
        note: The note to print.
        is_graph: Whether the model is a graph model.
        verbose: Whether to print the result.

    Returns:
        The test result.
    """
    num_batches = len(dataloader)

    if is_graph:
        torch.ao.quantization.move_exported_model_to_eval(model)
        # for n in model.graph.nodes:
        #     # Args: input, weight, bias, running_mean, running_var, training, momentum, eps
        #     # We set the `training` flag to False here to freeze BN stats
        #     # print(n.target)
        #     if n.target == torch.ops.aten.batch_norm.default:
        #         print(f"{list(n.args)}")
    else:
        model.eval()

    test_loss = 0
    r_errors = []

    # Time the model
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()

    preds = {}
    with torch.no_grad():
        for X, y, idx in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)

            # Exclude y_true of zero values
            mask = y != 0
            r_errors.append(torch.abs((y - pred) / y)[mask])

            # Save the prediction
            for i in range(len(idx)):
                key_name = idx[i].item()
                preds[key_name] = {}
                preds[key_name]["data"] = X[i].cpu().numpy()
                preds[key_name]["y_true"] = y[i].item() * 60.0
                preds[key_name]["y_pred"] = pred[i].item() * 60.0

    end.record()
    torch.cuda.synchronize()

    test_loss /= num_batches
    r_error = torch.cat(r_errors)
    ave_r_error = r_error.mean().item()
    max_r_error = r_error.max().item()
    std_r_error = r_error.std().item()

    if verbose:
        print(f"{note}_time (ms): {start.elapsed_time(end)} ms")
        print(f"{note}_ave_loss: {test_loss}")
        print(f"{note}_ave_relative_error: {ave_r_error * 100:.2f} %")
        print(f"{note}_max_relative_error: {max_r_error * 100:.2f} %")
        print(f"{note}_std_relative_error: {std_r_error * 100:.2f} %")

    result = {
        f"{note}_ave_loss": test_loss,
        f"{note}_ave_relative_error": ave_r_error,
        f"{note}_max_relative_error": max_r_error,
        f"{note}_std_relative_error": std_r_error,
        f"{note}_preds": preds,
    }
    return result


def test_pec(
    model: torch.nn.Module | torch.fx.GraphModule,
    loss_fn,
    device: str,
    is_graph: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Test the model with the PEC test set.

    Args:
        model: The model to be tested.
        loss_fn: The loss function.
        device: The device to run the model.
        is_graph: Whether the model is a graph model.
        verbose: Whether to print the result.

    Returns:
        The test result.
    """
    aluminum_test_dataloader = create_dataloader(
        load_json(Path("dataset") / "Aluminum_test.json"),
        data_length=256,
        batch_size=64,
        shuffle_flag=False,
    )
    aluminum_result = test(
        aluminum_test_dataloader,
        model,
        loss_fn,
        device,
        note="Aluminum",
        is_graph=is_graph,
        verbose=verbose,
    )

    s355_test_dataloader = create_dataloader(
        load_json(Path("dataset") / "S355_Stainless_Steel_test.json"),
        data_length=256,
        batch_size=64,
        shuffle_flag=False,
    )
    s355_result = test(
        s355_test_dataloader,
        model,
        loss_fn,
        device,
        note="S355_Stainless_Steel",
        is_graph=is_graph,
        verbose=verbose,
    )

    result = {**aluminum_result, **s355_result}
    return result
