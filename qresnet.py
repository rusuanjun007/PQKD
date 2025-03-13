from typing import Any, List, Optional, Type, Union

import torch
import torch.nn as nn
from torch import Tensor

from resnet import BasicBlock, Bottleneck, ResNet, _ovewrite_named_param

__all__ = [
    "QuantizableResNet",
    "qresnet18",
    "qresnet50",
]


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


def quantize_model(model: nn.Module, backend: str) -> None:
    _dummy_input_data = torch.rand(1, 1, 256)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == "fbgemm":
        model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.default_observer,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
    elif backend == "qnnpack":
        model.qconfig = torch.ao.quantization.QConfig(  # type: ignore[assignment]
            activation=torch.ao.quantization.default_observer,
            weight=torch.ao.quantization.default_weight_observer,
        )

    # TODO https://github.com/pytorch/vision/pull/4232#pullrequestreview-730461659
    model.fuse_model()  # type: ignore[operator]
    torch.ao.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.ao.quantization.convert(model, inplace=True)


class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(
            self, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], is_qat, inplace=True
        )
        if self.downsample:
            _fuse_modules(self.downsample, ["0", "1"], is_qat, inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)

        return out

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        _fuse_modules(
            self,
            [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["conv3", "bn3"]],
            is_qat,
            inplace=True,
        )
        if self.downsample:
            _fuse_modules(self.downsample, ["0", "1"], is_qat, inplace=True)


class QuantizableResNet(ResNet):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self, is_qat: Optional[bool] = None) -> None:
        r"""Fuse conv/bn/relu modules in resnet models

        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """
        _fuse_modules(self, ["conv1", "bn1", "relu"], is_qat, inplace=True)
        for m in self.modules():
            if type(m) is QuantizableBottleneck or type(m) is QuantizableBasicBlock:
                m.fuse_model(is_qat)


def _qresnet(
    block: Type[Union[QuantizableBasicBlock, QuantizableBottleneck]],
    layers: List[int],
    weights: Optional,
    progress: bool,
    quantize: bool,
    **kwargs: Any,
) -> QuantizableResNet:
    backend = kwargs.pop("backend", "fbgemm")

    model = QuantizableResNet(block, layers, **kwargs)
    _replace_relu(model)
    if quantize:
        quantize_model(model, backend)

    if weights is not None:
        model.load_state_dict(
            weights.get_state_dict(progress=progress, check_hash=True)
        )

    return model


def qresnet18(
    *,
    weights: Optional = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    """ResNet-18 model from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.ResNet18_QuantizedWeights` or :class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.ResNet18_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool, optional): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.ResNet18_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
        :noindex:
    """
    return _qresnet(
        QuantizableBasicBlock, [2, 2, 2, 2], weights, progress, quantize, **kwargs
    )


def qresnet34(
    *,
    weights: Optional = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    """ResNet-18 model from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.ResNet18_QuantizedWeights` or :class:`~torchvision.models.ResNet18_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.ResNet18_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool, optional): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.ResNet18_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.ResNet18_Weights
        :members:
        :noindex:
    """
    return _qresnet(
        QuantizableBasicBlock, [3, 4, 6, 3], weights, progress, quantize, **kwargs
    )


def qresnet50(
    *,
    weights: Optional = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> QuantizableResNet:
    """ResNet-50 model from
    `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_

    .. note::
        Note that ``quantize = True`` returns a quantized model with 8 bit
        weights. Quantized models only support inference and run on CPUs.
        GPU inference is not yet supported.

    Args:
        weights (:class:`~torchvision.models.quantization.ResNet50_QuantizedWeights` or :class:`~torchvision.models.ResNet50_Weights`, optional): The
            pretrained weights for the model. See
            :class:`~torchvision.models.quantization.ResNet50_QuantizedWeights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        quantize (bool, optional): If True, return a quantized version of the model. Default is False.
        **kwargs: parameters passed to the ``torchvision.models.quantization.QuantizableResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/quantization/resnet.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.quantization.ResNet50_QuantizedWeights
        :members:

    .. autoclass:: torchvision.models.ResNet50_Weights
        :members:
        :noindex:
    """

    return _qresnet(
        QuantizableBottleneck, [3, 4, 6, 3], weights, progress, quantize, **kwargs
    )


def qresnext50_32x4d(
    *,
    weights: Optional = None,
    progress: bool = True,
    quantize: bool = False,
    **kwargs: Any,
) -> ResNet:
    """ResNeXt-50 32x4d model from
    `Aggregated Residual Transformation for Deep Neural Networks <https://arxiv.org/abs/1611.05431>`_.

    Args:
        weights (:class:`~torchvision.models.ResNeXt50_32X4D_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.ResNext50_32X4D_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.resnet.ResNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ResNeXt50_32X4D_Weights
        :members:
    """

    _ovewrite_named_param(kwargs, "groups", 32)
    _ovewrite_named_param(kwargs, "width_per_group", 4)
    return _qresnet(Bottleneck, [3, 4, 6, 3], weights, progress, quantize, **kwargs)


if __name__ == "__main__":
    import onnx

    model = qresnet50(quantize=False)
    # model.fuse_model()

    X = torch.randn(66, 1, 256)

    model.eval()
    y = model(X)
    print(y.shape)

    # Save the model
    torch.save(model.state_dict(), "qresnet50.pth")

    if False:
        torch.onnx.export(model, X, "qresnet50.onnx")
        onnx.save(
            onnx.shape_inference.infer_shapes(onnx.load("qresnet50.onnx")),
            "qresnet50.onnx",
        )
