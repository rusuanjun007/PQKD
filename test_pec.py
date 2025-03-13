import numpy as np
import torch
from torch import nn

import model_baseline
import qmobilenetv3
import qresnet
from utils import test_pec


def test_float():
    def _test(float_model, weight_path):
        float_model.load_state_dict(
            torch.load(weight_path, map_location=device),
            strict=True,
        )
        float_model.eval()
        test_pec(float_model, loss_fn, device, is_graph=False)

    if False:
        float_model = model_baseline.MLP(256, 1840).to(device)
        weight_path = "model_float/MLP.pth"
        print("-" * 20 + "float mlp" + "-" * 20)
        _test(float_model, weight_path)

        float_model = model_baseline.CNN1D(1, 1130).to(device)
        weight_path = "model_float/CNN1D.pth"
        print("-" * 20 + "float cnn1d" + "-" * 20)
        _test(float_model, weight_path)

        float_model = model_baseline.LSTM(1, 440).to(device)
        weight_path = "model_float/LSTM.pth"
        print("-" * 20 + "float lstm" + "-" * 20)
        _test(float_model, weight_path)

        float_model = qresnet.qresnet18(num_classes=1).to(device)
        weight_path = "model_float/qresnet18.pth"
        print("-" * 20 + "float resnet18" + "-" * 20)
        _test(float_model, weight_path)

        float_model = qresnet.qresnet34(num_classes=1).to(device)
        weight_path = "model_float/qresnet34.pth"
        print("-" * 20 + "float resnet34" + "-" * 20)
        _test(float_model, weight_path)

        float_model = qresnet.qresnext50_32x4d(num_classes=1).to(device)
        weight_path = "model_float/qresnext50_32x4d.pth"
        print("-" * 20 + "float resnext50_32x4d" + "-" * 20)
        _test(float_model, weight_path)

        float_model = qmobilenetv3.qmobilenet_v3_small(num_classes=1).to(device)
        weight_path = "model_float/qmobilenetv3_small.pth"
        print("-" * 20 + "float mobilenet_v3_small" + "-" * 20)
        _test(float_model, weight_path)

        float_model = qmobilenetv3.qmobilenet_v3_large(num_classes=1).to(device)
        weight_path = "model_float/qmobilenetv3_large.pth"
        print("-" * 20 + "float mobilenet_v3_large" + "-" * 20)
        _test(float_model, weight_path)

    float_model = qresnet.qresnet50(num_classes=1).to(device)
    weight_path = "model_float/qresnet50.pth"
    print("-" * 20 + "float resnet50" + "-" * 20)
    _test(float_model, weight_path)


def test_pkd():
    def _test(weight_path):
        float_model = torch.load(weight_path, weights_only=False).to(device)
        float_model.eval()
        test_pec(float_model, loss_fn, device, is_graph=False)
        return float_model

    weight_path = "pkd/qresnet50_0.pth"
    print("-" * 20 + "pdk qresnet50_0" + "-" * 20)
    float_model = _test(weight_path)

    x = torch.randn(64, 1, 256).to(device)
    # Time the model
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )

    test_time = []
    with torch.no_grad():
        for _ in range(50):
            start.record()
            y = float_model(x)
            end.record()
            torch.cuda.synchronize()
            test_time.append(start.elapsed_time(end))

    print(test_time)
    print(f"Mean time: {np.mean(test_time)} ms")


if __name__ == "__main__":
    device = "cuda"
    loss_fn = nn.HuberLoss()

    # test_float()
    test_pkd()
