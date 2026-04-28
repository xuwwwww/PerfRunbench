from __future__ import annotations


def load_model_metadata(config: dict) -> dict:
    return config.get("model", {})


def load_torchvision_model(name: str):
    import torchvision.models as models

    model_factories = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "mobilenetv3": models.mobilenet_v3_small,
        "mobilenet_v3_small": models.mobilenet_v3_small,
    }
    if name not in model_factories:
        raise ValueError(f"Unsupported torchvision model: {name}")
    try:
        model = model_factories[name](weights=None)
    except TypeError:
        model = model_factories[name](pretrained=False)
    model.eval()
    return model
