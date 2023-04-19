from pathlib import Path
import json

def read_config(name, config_path = 'config.cfg') -> dict:    
    with open(config_path) as config_file:
        config_dict = json.load(config_file)
        # Check if config file has main mandatory keys
        assert all([i in config_dict.keys() for i in ['setup', 'service', 'model', 'train', 'eval']])
        return config_dict[name]

def get_path_stem(filepath): return str(Path(filepath).stem)

def get_path_name(filepath): return str(Path(filepath).name)

def get_path_parent(filepath): return str(Path(filepath).parent)

def join_path(root_dir, filepath): return str(Path(root_dir, filepath))

def get_file_extension(file): return str(Path(file).suffix)
