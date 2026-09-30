"""Convert Labelme polygon annotations to a YOLO segmentation dataset."""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / 'config.yaml'


@dataclass(frozen=True)
class Sample:
    """One Labelme annotation and its source image."""

    annotation_path: Path
    image_path: Path


def load_conversion_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the conversion section of config.yaml."""
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    with config_path.open(encoding='utf-8') as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(
        document.get('conversion'), dict
    ):
        raise ValueError('config.yaml must contain a conversion mapping')
    config = document['conversion']
    required = {
        'source_directory',
        'output_directory',
        'class_names',
        'train_ratio',
        'val_ratio',
        'test_ratio',
        'random_seed',
        'overwrite',
    }
    missing = sorted(required - set(config))
    unknown = sorted(set(config) - required)
    if missing:
        raise ValueError('Missing conversion settings: ' + ', '.join(missing))
    if unknown:
        raise ValueError('Unknown conversion settings: ' + ', '.join(unknown))
    class_names = config['class_names']
    if not isinstance(class_names, list) or not class_names:
        raise ValueError('class_names must be a non-empty list')
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError('Every class_names entry must be a non-empty string')
    if len(set(class_names)) != len(class_names):
        raise ValueError('class_names must not contain duplicates')
    for name in ('train_ratio', 'val_ratio', 'test_ratio'):
        if isinstance(config[name], bool) or not isinstance(
            config[name], (int, float)
        ):
            raise ValueError(f'{name} must be a number')
    validate_ratios(
        float(config['train_ratio']),
        float(config['val_ratio']),
        float(config['test_ratio']),
    )
    if isinstance(config['random_seed'], bool) or not isinstance(
        config['random_seed'], int
    ):
        raise ValueError('random_seed must be an integer')
    if not isinstance(config['overwrite'], bool):
        raise ValueError('overwrite must be boolean')
    return config


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    """Read one Labelme JSON document."""
    with path.open(encoding='utf-8') as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError(f'Labelme document must be an object: {path}')
    return document


def find_image(annotation_path: Path, document: dict[str, Any]) -> Path:
    """Resolve the image referenced by a Labelme annotation."""
    image_name = document.get('imagePath')
    candidates = []
    if image_name:
        candidates.append(annotation_path.parent / Path(image_name).name)
    # 用于尝试寻找与 JSON 文件同名的图像文件，支持多种常见图像格式
    candidates.extend(
        annotation_path.with_suffix(extension)
        for extension in ('.png', '.jpg', '.jpeg', '.bmp')
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'No image found for {annotation_path}')


def discover_samples(source: Path) -> list[Sample]:
    """Find all valid JSON/image pairs below the source directory."""
    samples = []
    for annotation_path in sorted(source.rglob('*.json')):
        document = load_json(annotation_path)
        samples.append(
            Sample(
                annotation_path=annotation_path,
                image_path=find_image(annotation_path, document),
            )
        )
    if not samples:
        raise FileNotFoundError(f'No Labelme JSON files found in {source}')
    return samples


def validate_ratios(train: float, val: float, test: float) -> None:
    """Validate dataset split ratios."""
    if min(train, val, test) < 0.0:
        raise ValueError('Split ratios cannot be negative')
    if abs(train + val + test - 1.0) > 1e-9:
        raise ValueError('train-ratio + val-ratio + test-ratio must equal 1')
    if train == 0.0 or val == 0.0:
        raise ValueError('Training and validation ratios must be greater than 0')


def split_samples(
    samples: list[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[Sample]]:
    """Shuffle samples reproducibly and assign dataset splits."""
    validate_ratios(train_ratio, val_ratio, test_ratio)
    # 将样本进行浅拷贝，用于进行顺序打算，仅需要打乱引用索引即可，所以不需要进行深拷贝
    shuffled = samples.copy()
    random.Random(seed).shuffle(shuffled)
    sample_count = len(shuffled)
    train_end = round(sample_count * train_ratio)
    val_end = train_end + round(sample_count * val_ratio)
    splits = {
        'train': shuffled[:train_end],
        'val': shuffled[train_end:val_end],
    }
    if test_ratio > 0.0:
        splits['test'] = shuffled[val_end:]
    if not splits['train'] or not splits['val']:
        raise ValueError('Not enough samples for non-empty train and val splits')
    return splits


def normalize_polygon(
    points: Any, image_width: int, image_height: int
) -> list[float]:
    """Validate, clip and normalize one polygon to the [0, 1] range."""
    if not isinstance(points, list) or len(points) < 3:
        raise ValueError('A segmentation polygon requires at least 3 points')
    normalized = []
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f'Invalid polygon point: {point!r}')
        x_coordinate = min(max(float(point[0]), 0.0), float(image_width))
        y_coordinate = min(max(float(point[1]), 0.0), float(image_height))
        normalized.extend(
            (x_coordinate / image_width, y_coordinate / image_height)
        )
    return normalized


# 为了防止在labelme中用了rectangle而不是polygon，增加一个转换函数
def rectangle_to_polygon(points: Any) -> list[list[float]]:
    """Convert two opposite Labelme rectangle corners to four vertices."""
    if not isinstance(points, list) or len(points) != 2:
        raise ValueError('A rectangle must contain exactly 2 points')
    x1, y1 = (float(value) for value in points[0])
    x2, y2 = (float(value) for value in points[1])
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def convert_annotation(
    annotation_path: Path,
    class_ids: dict[str, int],
) -> list[str]:
    """Convert matching Labelme shapes into YOLO segmentation rows."""
    document = load_json(annotation_path)
    image_width = int(document.get('imageWidth', 0))
    image_height = int(document.get('imageHeight', 0))
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f'Invalid image dimensions in {annotation_path}')

    rows = []
    for shape in document.get('shapes', []):
        source_label = str(shape.get('label', ''))
        if source_label not in class_ids:
            continue
        shape_type = shape.get('shape_type', 'polygon')
        points = shape.get('points')
        if shape_type == 'rectangle':
            points = rectangle_to_polygon(points)
        elif shape_type != 'polygon':
            raise ValueError(
                f'Unsupported shape type {shape_type!r} in {annotation_path}'
            )
        coordinates = normalize_polygon(points, image_width, image_height)
        rows.append(
            f'{class_ids[source_label]} '
            + ' '.join(f'{value:.8f}' for value in coordinates)
        )
    if not rows:
        raise ValueError(
            f'No configured class labels found in {annotation_path}'
        )
    return rows


def prepare_output(output: Path, source: Path, overwrite: bool) -> None:
    """Create an empty output directory without accidental data loss."""
    source = source.resolve()
    if output.resolve() == source or output.resolve() in source.parents:
        raise ValueError(
            'Output cannot be the source dataset or one of its parent folders'
        )
    if output.exists():
        if not overwrite:
            raise FileExistsError(
                f'Output already exists: {output}; set conversion.overwrite '
                'to true in config.yaml to replace it'
            )
        shutil.rmtree(output)
    output.mkdir(parents=True)


def write_dataset(
    splits: dict[str, list[Sample]],
    output: Path,
    class_names: list[str],
) -> None:
    """Copy split images, write labels and generate data.yaml."""
    class_ids = {name: index for index, name in enumerate(class_names)}
    for split_name, samples in splits.items():
        image_directory = output / 'images' / split_name
        label_directory = output / 'labels' / split_name
        image_directory.mkdir(parents=True)
        label_directory.mkdir(parents=True)
        used_names = set()
        for sample in samples:
            output_name = sample.image_path.name
            if output_name in used_names:
                raise ValueError(f'Duplicate image filename: {output_name}')
            used_names.add(output_name)
            rows = convert_annotation(sample.annotation_path, class_ids)
            shutil.copy2(sample.image_path, image_directory / output_name)
            label_path = label_directory / f'{sample.image_path.stem}.txt'
            label_path.write_text('\n'.join(rows) + '\n', encoding='utf-8')

    # data.yaml文件不需要指定labels目录，因为YOLO会自动将images/train等路径替换成labels，寻找对应标签
    yaml_lines = [
        f'path: {output.resolve()}',
        'train: images/train',
        'val: images/val',
    ]
    if 'test' in splits:
        yaml_lines.append('test: images/test')
    yaml_lines.extend(('', 'names:'))
    yaml_lines.extend(
        f'  {index}: {name}' for index, name in enumerate(class_names)
    )
    yaml_lines.append('')
    (output / 'data.yaml').write_text('\n'.join(yaml_lines), encoding='utf-8')


def main() -> None:
    """Convert the configured Labelme dataset."""
    config = load_conversion_config(CONFIG_PATH)
    source = resolve_project_path(config['source_directory'])
    output = resolve_project_path(config['output_directory'])
    samples = discover_samples(source)
    splits = split_samples(
        samples,
        float(config['train_ratio']),
        float(config['val_ratio']),
        float(config['test_ratio']),
        int(config['random_seed']),
    )
    prepare_output(output, source, bool(config['overwrite']))
    write_dataset(
        splits,
        output,
        list(config['class_names']),
    )
    counts = ', '.join(
        f'{name}={len(items)}' for name, items in splits.items()
    )
    print(f'Converted {len(samples)} images: {counts}')
    print(f'Dataset YAML: {output / "data.yaml"}')


if __name__ == '__main__':
    main()
