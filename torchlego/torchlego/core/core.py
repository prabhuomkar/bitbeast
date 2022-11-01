"""TorchLego Core"""
import os
from typing import Dict, Any
import logging
import requests

from torch.jit import load

from config import process_yaml_config, ModelConfig
from core.input import derive_input
from core.preprocess import derive_preprocess

# this will store the derived model ready for execution
MODELS = {}

def init_models() -> None:
    """Derive model by mapping config to executable functions"""
    model_cfg_file = os.getenv("MODEL_CONFIG", "models.yaml")
    with open(model_cfg_file, encoding="UTF-8") as yaml_file:
        file_contents = yaml_file.read()
    cfg = process_yaml_config(file_contents)
    logging.debug("derived yaml config: %s", cfg)
    for model in cfg.models:
        config = model.dict()
        MODELS[model.name] = dict({"config": config})
    for model in cfg.models:
        executable = create_executable(model)
        if executable is not None:
            MODELS[model.name]["config"]["initialized"] = True
            MODELS[model.name]["executable"] = executable
            logging.info("created executable for model: %s", model.name)
        else:
            logging.error("error creating executable for model: %s", model.name)

def get_models() -> Dict[Any, Any]:
    """Get derived models"""
    models = []
    for data in MODELS.values():
        models.append(data["config"])
    return models

def download_model(model_name: str, link: str) -> bool:
    """Download model for inference"""
    try:
        logging.info("downloading model: %s", model_name)
        response = requests.get(link)
        with open(f"{model_name}", "wb") as model_file:
            model_file.write(response.content)
        logging.info("finished downloading model: %s", model_name)
        return True
    except:
        logging.error("error downloading model: %s url: %s", model_name, link)
    return False

def create_executable(model: ModelConfig) -> Any:
    """Create runtime executable from stages"""
    downloaded = download_model(model.name, model.download)
    if downloaded:
        model_input = derive_input(model.stages.input)
        preprocess = derive_preprocess(model.stages.preprocess)
        inference = load(f"{model.name}")
        return [model_input] + preprocess + [inference]
    return None

def run_executable(model_name: str, request: Any) -> bytes:
    """Run executable to get output from derived model"""
    if model_name in MODELS:
        model = MODELS[model_name]
        if not model["config"]["initialized"]:
            raise Exception("model not initialized")
        executable = model["executable"]
        result = executable[0](request)
        for idx in range(1, len(executable)):
            execute = executable[idx]
            result = execute(result)
        return result
    raise Exception("model not found")
