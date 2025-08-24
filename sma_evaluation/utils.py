import yaml
import json
from typing import Dict, Any


# Support Functions
def read_json(file_path: str) -> Dict[str, Any]:
    """Read a JSON file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: The contents of the JSON file as a dictionary.
    """
    with open(file_path, encoding="utf8") as f:
        return json.load(f)

def read_yaml(path):
    """
    Read and parse a YAML file from the given path.

    Args:
        path (str): The file path to the YAML file.

    Returns:
        dict: The parsed YAML content as a dictionary. Returns an empty dict if parsing fails.
    """
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return {}