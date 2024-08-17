
from src.training import train_model


import yaml
from pathlib import Path

def load_config(config_path='resources/config.yaml'):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    config = load_config()
    train_model(config)
