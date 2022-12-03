"""YAML Config"""
from ast import Str
from typing import Any, Dict, List
from pydantic import BaseModel

import yaml


class StageConfig(BaseModel):
    """Stage Config"""
    input: str
    preprocess: Dict[Any, Any]


class ModelConfig(BaseModel):
    """Model Config"""
    name: str
    download: str
    gpu: bool = False
    initialized: bool = False
    stages: StageConfig


class YAMLConfig(BaseModel):
    """YAML Config"""
    models: List[ModelConfig]


def process_yaml_config(file_contents: Str) -> YAMLConfig:
    """read the yaml file and parse config fields"""
    cfg_map = yaml.safe_load(file_contents)
    return YAMLConfig(**cfg_map)
