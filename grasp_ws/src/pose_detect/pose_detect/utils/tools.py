"""Shared configuration loading helpers."""

from pathlib import Path

import yaml


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a relative path from the project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / '.gitmodules').is_file():
            return (parent / path).resolve()
    raise FileNotFoundError('Project root containing .gitmodules was not found')


def _flatten_config(mapping: dict, node_name: str, prefix: str = '') -> dict:
    """Flatten nested mappings into dotted parameter names."""
    config = {}
    for key, value in mapping.items():
        if not isinstance(key, str) or not key or '.' in key:
            raise ValueError(
                f'{node_name}: configuration keys must be nonempty strings without dots'
            )
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            if not value:
                raise ValueError(f'{node_name}.{name} must not be an empty group')
            config.update(_flatten_config(value, node_name, name))
        else:
            config[name] = value
    return config


'''
加载节点配置，将嵌套字典展开为 ROS 参数名（如 general.base_frame）。
@param config_path: YAML 配置文件路径
@param node_name: 节点名称，必须是配置文件的顶级键
@param config_params: 期望的完整展开参数名集合；None 表示不校验参数名。列表保留为参数值，配置键不允许包含点号，以避免展开后的名称冲突。
@param path_params: 需要相对于项目根目录解析的参数名集合
@return: 展开后的参数字典
'''
def load_config(
    config_path: str | Path,
    node_name: str,
    config_params: set[str] | None = None,
    path_params: set[str] | None = None,
) -> dict:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f'Configuration file not found: {config_path}')
    with config_path.open(encoding='utf-8') as stream:
        document = yaml.safe_load(stream)
    try:
        section = document[node_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f'{config_path} must contain {node_name}') from error
    if not isinstance(section, dict):
        raise ValueError(f'{node_name} must be a YAML mapping')

    config = _flatten_config(section, node_name)
    if config_params is not None:
        missing = sorted(config_params - set(config))
        unknown = sorted(set(config) - config_params)
        if missing:
            raise ValueError(
                f'{node_name}: missing parameters: ' + ', '.join(missing)
            )
        if unknown:
            raise ValueError(
                f'{node_name}: unknown parameters: ' + ', '.join(unknown)
            )
    for name in path_params or ():
        config[name] = str(resolve_project_path(config[name]))
    return config
