import os
import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")

with open(_CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

models = _config["models"]
