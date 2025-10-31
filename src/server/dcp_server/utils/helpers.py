from pathlib import Path
import yaml


def read_config(name: str, config_path: str) -> dict:
    """Reads the configuration file

    :param name: name of the section you want to read (e.g. 'setup', 'eval')
    :type name: string
    :param config_path: path to the configuration file
    :type config_path: str
    :return: dictionary from the config section given by name
    :rtype: dict
    """
    with open(config_path) as config_file:
        config_dict = yaml.safe_load(
            config_file
        )  # json.load(config_file) for .cfg file
        # Check if config file has main mandatory keys
        assert all(
            [
                i in config_dict.keys()
                for i in ["setup", "service", "model", "eval"]
            ]
        )
        return config_dict[name]


def get_path_stem(filepath: str) -> str:
    return str(Path(filepath).stem)


def get_path_name(filepath: str) -> str:
    return str(Path(filepath).name)


def get_path_parent(filepath: str) -> str:
    return str(Path(filepath).parent)


def join_path(root_dir: str, filepath: str) -> str:
    return str(Path(root_dir, filepath))


def get_file_extension(file: str) -> str:
    return str(Path(file).suffix)
