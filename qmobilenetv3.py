from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor, nn
from torch.ao.quantization import DeQuantStub, QuantStub

from mobilenetv3 import (
    Conv1dNormActivation,
    InvertedResidual,
    InvertedResidualConfig,
    MobileNetV3,
    SElayer,
    _mobilenet_v3_conf,
)


def _ovewrite_named_param(kwargs: Dict[str, Any], param: str, new_value) -> None:
    if param in kwargs:
        if kwargs[param] != new_value:
            raise ValueError(
                f"The parameter '{param}' expected value {new_value} but got {kwargs[param]} instead."
            )
    else:
        kwargs[param] = new_value


def _replace_relu(module: nn.Module) -> None:
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) is nn.ReLU or type(mod) is nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def _fuse_modules(
    model: nn.Module,
    modules_to_fuse: Union[List[str], List[List[str]]],
    is_qat: Optional[bool],
    **kwargs: Any,
):
    if is_qat is None:
        is_qat = model.training
    method = (
        torch.ao.quantization.fuse_modules_qat
        if is_qat
        else torch.ao.quantization.fuse_modules
    )
    return method(model, modules_to_fuse, **kwargs)


__all__ = [
    "QuantizableMobileNetV3",
    "qmobilenet_v3_large",
]


class QuantizableSElayer(SElayer):
    _version = 2

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["scale_activation"] = nn.Hardsigmoid
        super().__init__(*args, **kwargs)
        self.skip_mul = nn.quantized.FloatFunctional()

    def forward(self, input: Tensor) -> Tensor:
        return self.skip_mul.mul(self._scale(input), input)

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(self, ["fc1", "activation"], is_qat, inplace=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if hasattr(self, "qconfig") and (version is None or version < 2):
            default_state_dict = {
                "scale_activation.activation_post_process.scale": torch.tensor([1.0]),
                "scale_activation.activation_post_process.activation_post_process.scale": torch.tensor(
                    [1.0]
                ),
                "scale_activation.activation_post_process.zero_point": torch.tensor(
                    [0], dtype=torch.int32
                ),
                "scale_activation.activation_post_process.activation_post_process.zero_point": torch.tensor(
                    [0], dtype=torch.int32
                ),
                "scale_activation.activation_post_process.fake_quant_enabled": torch.tensor(
                    [1]
                ),
                "scale_activation.activation_post_process.observer_enabled": torch.tensor(
                    [1]
                ),
            }
            for k, v in default_state_dict.items():
                full_key = prefix + k
                if full_key not in state_dict:
                    state_dict[full_key] = v

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class QuantizableInvertedResidual(InvertedResidual):
    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, se_layer=QuantizableSElayer, **kwargs)  # type: ignore[misc]
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return self.skip_add.add(x, self.block(x))
        else:
            return self.block(x)


class QuantizableMobileNetV3(MobileNetV3):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        MobileNet V3 main class

        Args:
           Inherits args from floating point MobileNetV3
        """
        super().__init__(*args, **kwargs)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        for m in self.modules():
            if type(m) is Conv1dNormActivation:
                modules_to_fuse = ["0", "1"]
                if len(m) == 3 and type(m[2]) is nn.ReLU:
                    modules_to_fuse.append("2")
                _fuse_modules(m, modules_to_fuse, is_qat, inplace=True)
            elif type(m) is QuantizableSElayer:
                m.fuse_model(is_qat)


def _qmobilenet_v3_model(
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    weights: Optional,
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        if "backend" in weights.meta:
            _ovewrite_named_param(kwargs, "backend", weights.meta["backend"])
    backend = kwargs.pop("backend", "qnnpack")

    model = QuantizableMobileNetV3(
        inverted_residual_setting,
        last_channel,
        block=QuantizableInvertedResidual,
        **kwargs,
    )
    _replace_relu(model)

    if quantize:
        # Instead of quantizing the model and then loading the quantized weights we take a different approach.
        # We prepare the QAT model, load the QAT weights from training and then convert it.
        # This is done to avoid extremely low accuracies observed on the specific model. This is rather a workaround
        # for an unresolved bug on the eager quantization API detailed at: https://github.com/pytorch/vision/issues/5890
        model.fuse_model(is_qat=True)
        model.qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
        torch.ao.quantization.prepare_qat(model, inplace=True)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    if quantize:
        torch.ao.quantization.convert(model, inplace=True)
        model.eval()

    return model


def qmobilenet_v3_large(
    *,
    weights: Optional = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    """
    MobileNetV3 (Large) model from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`_.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights` or :class:`~torchvision.models.MobileNet_V3_Large_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.MobileNet_V3_Large_QuantizedWeights
        :members:
    .. autoclass:: torchvision.models.MobileNet_V3_Large_Weights
        :members:
        :noindex:
    """

    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_large", **kwargs
    )
    return _qmobilenet_v3_model(
        inverted_residual_setting, last_channel, weights, progress, quantize, **kwargs
    )


def qmobilenet_v3_small(
    *,
    weights: Optional = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableMobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `Searching for MobileNetV3 <https://arxiv.org/abs/1905.02244>`__.

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.MobileNet_V3_Small_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.MobileNet_V3_Small_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.mobilenet.MobileNetV3``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.MobileNet_V3_Small_Weights
        :members:
    """

    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        "mobilenet_v3_small", **kwargs
    )
    return _qmobilenet_v3_model(
        inverted_residual_setting, last_channel, weights, progress, quantize, **kwargs
    )


if __name__ == "__main__":
    import onnx

    model = qmobilenet_v3_large(quantize=False)

    X = torch.randn(66, 1, 256)

    model.eval()
    y = model(X)
    print(y.shape)
    torch.save(model.state_dict(), "qmobilenet_v3_large.pth")

    if False:
        torch.onnx.export(model, X, "qmobilenet_v3_small.onnx")
        onnx.save(
            onnx.shape_inference.infer_shapes(onnx.load("qmobilenet_v3_small.onnx")),
            "qmobilenet_v3_small.onnx",
        )
