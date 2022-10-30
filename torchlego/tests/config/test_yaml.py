
"""Config YAML Tests"""
from torchlego.config import YAMLConfig, process_yaml_config


FILE_CONTENTS = '''
models:
  - name: resnet50
    download: https://artifactory/model-download-link
    gpu: true
    stages:
      input: file
      preprocess: 
        default: image_classification
  - name: resnet50
    download: https://artifactory/model-download-link
    gpu: true
    stages:
      input: file
      preprocess:
        resize: 299
        center_crop: 299
        to_tensor: true
        normalize:
          mean: [0.485, 0.456, 0.406]
          std: [0.229, 0.224, 0.225]
'''

def test_process_yaml_config():
    yaml_cfg = process_yaml_config(FILE_CONTENTS)
    assert type(yaml_cfg) == YAMLConfig
    assert len(yaml_cfg.models) == 2
    assert yaml_cfg.models[1].name == 'resnet50'
    assert yaml_cfg.models[1].download == 'https://artifactory/model-download-link'
    assert yaml_cfg.models[1].gpu == True
    assert yaml_cfg.models[1].stages.input == "file"
    assert len(yaml_cfg.models[1].stages.preprocess.items()) == 4
