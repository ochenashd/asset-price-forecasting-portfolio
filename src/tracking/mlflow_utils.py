# src/tracking/mlflow_utils.py

from pathlib import Path
from typing import Any, Dict

import mlflow
import yaml
from omegaconf import DictConfig, OmegaConf


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten a nested dictionary for MLflow parameter logging."""
    items = {}

    for key, value in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value

    return items


def log_model_params_to_mlflow(
    config: DictConfig,
    model_name: str,
    output_dir: str = "results/reports/mlflow_params",
) -> None:
    """Log model parameters to MLflow as params and as a YAML artifact."""
    if mlflow.active_run() is None:
        return

    models_config = config.get("models", {})

    if model_name not in models_config:
        mlflow.log_param(f"{model_name}.config_found", False)
        return

    model_params = models_config.get(model_name, {})

    if isinstance(model_params, DictConfig):
        model_params = OmegaConf.to_container(
            model_params,
            resolve=True,
        )

    model_params = dict(model_params)

    flat_params = flatten_dict(model_params)

    for key, value in flat_params.items():
        mlflow.log_param(f"{model_name}.{key}", value)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    yaml_path = output_path / f"{model_name}_params.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            model_params,
            file,
            sort_keys=False,
            allow_unicode=True,
        )

    mlflow.log_artifact(
        str(yaml_path),
        artifact_path="model_params",
    )