"""Background dataset-generation job."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import dataset_templates


def run(emit: Callable[[dict], None], config: dict) -> None:
    template_id = config["template_id"]
    inputs = config.get("inputs") or {}
    output_dir = Path(config["output_dir"])

    emit({"type": "stage", "stage": "starting",
          "message": f"Starting template {template_id}…", "progress": 0.05})

    dataset_templates.generate(
        template_id=template_id,
        inputs=inputs,
        output_dir=output_dir,
        emit=emit,
        api_url=config.get("api_url"),
        auth_token=config.get("auth_token"),
        model_name=config.get("model_name"),
    )
