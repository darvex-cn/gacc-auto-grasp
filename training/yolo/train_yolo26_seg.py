"""Train YOLO26m-seg on the converted motor-part dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_training_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the training section of config.yaml."""
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    with config_path.open(encoding='utf-8') as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(
        document.get('training'), dict
    ):
        raise ValueError('config.yaml must contain a training mapping')
    config = document['training']
    required = {
        'model_path',
        'data_yaml',
        'epochs',
        'image_size',
        'batch_size',
        'device',
        'workers',
        'patience',
        'cache',
        'resume',
        'project_directory',
        'run_name',
    }
    missing = sorted(required - set(config))
    unknown = sorted(set(config) - required)
    if missing:
        raise ValueError('Missing training settings: ' + ', '.join(missing))
    if unknown:
        raise ValueError('Unknown training settings: ' + ', '.join(unknown))
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail before training when paths or numeric options are invalid."""
    model_path = resolve_project_path(config['model_path'])
    data_yaml = resolve_project_path(config['data_yaml'])
    if not model_path.is_file():
        raise FileNotFoundError(f'Model weights not found: {model_path}')
    if not data_yaml.is_file():
        raise FileNotFoundError(f'Dataset YAML not found: {data_yaml}')
    for name in ('epochs', 'image_size', 'batch_size'):
        value = int(config[name])
        if value <= 0:
            raise ValueError(f'{name} must be greater than zero')
    if int(config['workers']) < 0:
        raise ValueError('workers must be non-negative')
    if int(config['patience']) < 0:
        raise ValueError('patience must be non-negative')
    if not str(config['device']).strip():
        raise ValueError('device must not be empty')
    if not str(config['run_name']).strip():
        raise ValueError('run_name must not be empty')
    for name in ('cache', 'resume'):
        if not isinstance(config[name], bool):
            raise ValueError(f'{name} must be boolean')


def main() -> None:
    """Load pretrained weights and start segmentation training."""
    config = load_training_config(CONFIG_PATH)
    validate_config(config)
    model_path = resolve_project_path(config['model_path'])
    data_yaml = resolve_project_path(config['data_yaml'])
    project_directory = resolve_project_path(config['project_directory'])
    model = YOLO(str(model_path))
    result = model.train(
        data=str(data_yaml),
        epochs=int(config['epochs']),
        imgsz=int(config['image_size']),
        batch=int(config['batch_size']),
        device=str(config['device']),
        workers=int(config['workers']),
        patience=int(config['patience']),
        project=str(project_directory),
        name=str(config['run_name']),
        cache=config['cache'],
        resume=config['resume'],
    )
    save_directory = getattr(result, 'save_dir', None)
    print(f'Training finished. Results: {save_directory}')


if __name__ == '__main__':
    main()
