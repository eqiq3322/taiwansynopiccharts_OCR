from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def resolve_path(value: Optional[str], project_root: Path) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_config_section(config_path: str, section: str) -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return {}

    section_cfg = data.get(section, {})
    if not isinstance(section_cfg, dict):
        section_cfg = {}

    # Merge shared top-level paths into section defaults
    if "img_folder" in data and "img_folder" not in section_cfg:
        section_cfg["img_folder"] = data["img_folder"]
    if "data_folder" in data and "data_folder" not in section_cfg:
        section_cfg["data_folder"] = data["data_folder"]

    return section_cfg
