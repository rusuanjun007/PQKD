import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Set random seed.
np.random.seed(666)
torch.manual_seed(667)


def load_json(json_path: Path) -> dict:
    """Load data from json file.

    Args:
        json_path (Path): The path of the json file.

    Raises:
        FileNotFoundError:File do not exist.

    Returns:
        dict: The data in the json file.
    """

    if json_path.exists():
        # print(f"Load data form {json_path}")
        with open(json_path, "r") as file:
            data = json.load(file)
        return data
    else:
        raise FileNotFoundError(f"{json_path} do not exist.")


def label_filter(data_dict: dict, label_dict: dict, verbose: bool = False) -> list:
    """Filter the data with labels.

    Labels:
    "Thickness" in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    "Lift-off" in [0, 3, 6, 9, 12, 15]
    "Loc" in [Center, Corner, Edge, Random]
    "Insulation" [0, 5, 10, 15]
    "WeatherJacket" in [0, 3]

    Args:
        data_dict (dict): The data dictionary.
        label_dict (dict): The label dictionary.
        verbose (bool, optional): Whether to print the number of targets. Defaults to False.

    Returns:
        list: The list of keys of the filtered data.
    """
    targets = []
    for key, value in data_dict.items():
        temp_flag = True
        for label in label_dict:
            if label_dict[label] != value["label"][label]:
                temp_flag = False
                break
        if temp_flag:
            targets.append(key)
    if verbose:
        print(f"For {label_dict}, found {len(targets)} targets")
    return targets


def plot_filtered_data(
    dataset: dict, label_dicts: list, key: str, root_path: Path = None
):
    """Plot the data, for same label use same color, only one legend for each label

    Args:
        dataset (dict): The data dictionary.
        label_dicts (list): The list of label dictionaries for filtering.
        key (str):  The key for plot legend.
        root_path (Path, optional): The root path for saving the figure. Defaults to None.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    legends = []
    for i, label_dict in enumerate(label_dicts):
        targets = label_filter(dataset, label_dict)

        for target in targets:
            data = dataset[target]["data"]
            legend = dataset[target]["label"][key]

            if f"{legend} mm" not in legends:
                legends.append(f"{legend} mm")
                ax.plot(data[:256], label=f"{legend} mm", color=f"C{i}")
            else:
                ax.plot(data[:256], color=f"C{i}")

    ax.legend()
    plt.tight_layout()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (V)")

    if root_path:
        name = ""
        for label in label_dicts:
            name += f"Thickness{label['Thickness']}_Lift{label['Lift-off']}_Loc{label['Loc']}_Insulation{label['Insulation']}_WeatherJacket{label['WeatherJacket']}"
        plt.savefig(root_path / name)
        plt.close()
    else:
        plt.show()


def summary_dataset(aluminum_dict: dict, s355_dict: dict):
    """Summary the aluminum and s355 dataset.

    Args:
        aluminum_dict (dict): the aluminum dataset.
        s355_dict (dict): the S355 dataset.
    """

    print("Aluminum")
    data_max, data_min = 0, 999999
    count = 0
    for thickness in range(20, 65, 5):
        for lift_off in range(0, 18, 3):
            for loc in ["Center", "Corner", "Edge", "Random"]:
                label_dicts = {
                    "Thickness": thickness,
                    "Lift-off": lift_off,
                    "Loc": loc,
                    "Insulation": 0,
                    "WeatherJacket": 0,
                }
                filtered_keys = label_filter(aluminum_dict, label_dicts)
                count += len(filtered_keys)
                for key in filtered_keys:
                    data_max = max(data_max, max(aluminum_dict[key]["data"]))
                    data_min = min(data_min, min(aluminum_dict[key]["data"]))
                # plot_filtered_data(
                #     aluminum_dict, [label_dicts], "Thickness", Path("aluminum")
                # )
    print(f"Aluminum Max: {data_max}, Min: {data_min}, Count: {count}")

    # Summary each label data
    print("S355")
    data_max, data_min = 0, 999999
    count = 0
    for thickness in range(0, 25, 5):
        for lift_off in range(0, 12, 3):
            for insulation in range(0, 20, 5):
                for weather_jacket in range(0, 6, 3):
                    label_dicts = {
                        "Thickness": thickness,
                        "Lift-off": lift_off,
                        "Loc": "Center",
                        "Insulation": insulation,
                        "WeatherJacket": weather_jacket,
                    }
                    filtered_keys = label_filter(s355_dict, label_dicts, verbose=True)
                    count += len(filtered_keys)
                    for key in filtered_keys:
                        data_max = max(data_max, max(s355_dict[key]["data"]))
                        data_min = min(data_min, min(s355_dict[key]["data"]))
                    # plot_filtered_data(
                    #     s355_dict, [label_dicts], "Thickness", Path("s355")
                    # )
    print(f"S355 Max: {data_max}, Min: {data_min}, Count: {count}")


def preprocess(
    data_dict: dict,
    name: str,
    split_rate: float = 0.8,
    shuffle_flag: bool = True,
) -> list:
    """
    Preprocess the data and save the train, validation and test data as json file.

    Args:
        data_dict (dict): The data dictionary.
        name (str): The name of the dataset.
        split_rate (float, optional): The split rate for train, validation and test. Defaults to 0.8.
        shuffle_flag (bool, optional): Whether to shuffle the data. Defaults

    Returns:
        list: The list of keys of the filtered data.
        [TrainDataPipeline, ValidationDataPipeline, TestDataPipeline]
    """

    # Set random seed.
    np.random.seed(666)

    material_list = ["Aluminum", "S355"]
    thickness_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    lift_off_list = [0, 3, 6, 9, 12, 15]
    loc_list = ["Center", "Corner", "Edge", "Random"]
    insulation_list = [0, 5, 10, 15]
    weather_jacket_list = [0, 3]

    train_key, val_key, test_key = [], [], []
    for material in material_list:
        for thickness in thickness_list:
            for lift_off in lift_off_list:
                for loc in loc_list:
                    for insulation in insulation_list:
                        for weather_jacket in weather_jacket_list:
                            label_dicts = {
                                "Material": material,
                                "Thickness": thickness,
                                "Lift-off": lift_off,
                                "Loc": loc,
                                "Insulation": insulation,
                                "WeatherJacket": weather_jacket,
                            }
                            filtered_keys = label_filter(data_dict, label_dicts)
                            if len(filtered_keys) == 0:
                                continue

                            if shuffle_flag:
                                np.random.shuffle(filtered_keys)
                            split_idx = int(len(filtered_keys) * split_rate)
                            n_data = len(filtered_keys)

                            # Split is for train, rest is half for validation and half for test
                            # For example, if split_rate = 0.8, then 80% for train, 10% for validation, 10% for test
                            train_key.extend(filtered_keys[:split_idx])
                            val_key.extend(
                                filtered_keys[
                                    split_idx : split_idx + (n_data - split_idx) // 2
                                ]
                            )
                            test_key.extend(
                                filtered_keys[split_idx + (n_data - split_idx) // 2 :]
                            )

    train_data = {key: data_dict[key] for key in train_key}
    val_data = {key: data_dict[key] for key in val_key}
    test_data = {key: data_dict[key] for key in test_key}

    # Save the data as json
    with open(data_root_path / (name + "_train.json"), "w") as f:
        json.dump(train_data, f)
    with open(data_root_path / (name + "_val.json"), "w") as f:
        json.dump(val_data, f)
    with open(data_root_path / (name + "_test.json"), "w") as f:
        json.dump(test_data, f)


def create_dataloader(
    data_dict: dict,
    data_length: int,
    batch_size: int,
    shuffle_flag: bool,
) -> torch.utils.data.DataLoader:
    """Create dataloader for the data.

    Args:
        data_dict (dict): The data dictionary.
        data_length (int): The length of the data.
        batch_size (int): The batch size.
        shuffle_flag (bool): Whether to shuffle the data.

    Returns:
        torch.utils.data.DataLoader: The dataloader.
    """
    # Normalize the data with max_data from train set, for S355 and Aluminum
    max_data = 3799.0
    max_thickness = 60.0

    # Create dataset
    # Data shape: [n_data, n_channel=1, data_length]
    data = torch.tensor(
        [data_dict[key]["data"][0:data_length] for key in data_dict.keys()],
        dtype=torch.float32,
    )
    data = data / max_data
    data = data.view(-1, 1, data_length)

    # Label shape: [n_data, 1]
    label = torch.tensor(
        [data_dict[key]["label"]["Thickness"] for key in data_dict.keys()],
        dtype=torch.float32,
    )
    label = label / max_thickness
    label = label.view(-1, 1)

    # idx: [n_data]
    idx = torch.tensor(
        [data_dict[key]["label"]["idx"] for key in data_dict.keys()],
        dtype=torch.int32,
    )

    dataset = torch.utils.data.TensorDataset(data, label, idx)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_flag, drop_last=True
    )

    return dataloader


if __name__ == "__main__":
    data_root_path = Path("dataset")
    aluminum_dict = load_json(data_root_path / "Aluminum.json")
    s355_dict = load_json(data_root_path / "S355_Stainless_Steel.json")

    # summary_dataset()
    # preprocess(aluminum_dict, name="Aluminum")
    # preprocess(s355_dict, name="S355_Stainless_Steel")

    label_dicts = []
    label_dicts = label_dicts + [
        {
            "Thickness": i,
            "Lift-off": 0,
            "Loc": "Center",
            "Insulation": 0,
            "WeatherJacket": 0,
        }
        for i in range(20, 65, 5)
    ]
    # plot_filtered_data(aluminum_dict, label_dicts, "Thickness")

    label_dicts = []
    label_dicts = label_dicts + [
        {
            "Thickness": i,
            "Lift-off": 0,
            "Loc": "Center",
            "Insulation": 0,
            "WeatherJacket": 0,
        }
        for i in range(0, 25, 5)
    ]
    # plot_filtered_data(s355_dict, label_dicts, "Thickness")

    # Create dataloader
    aluminum_train = create_dataloader(
        load_json(data_root_path / "Aluminum_train.json"),
        data_length=256,
        batch_size=32,
        shuffle_flag=True,
    )
    for data in aluminum_train:
        print(data[0].shape, data[0].max(), data[0].min())
        print(data[1])
        print(data[2])
        break

    for data in aluminum_train:
        print(data[0].shape, data[0].max(), data[0].min())
        print(data[1])
        print(data[2])
        break
