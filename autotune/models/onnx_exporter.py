from __future__ import annotations

from pathlib import Path


def export_to_onnx(model, input_shape: list[int], output_path: str | Path) -> Path:
    import torch

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(
        model,
        dummy_input,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
    )
    return path
