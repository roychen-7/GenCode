import json
import os


# --------------------------
# Utility: load JSON
# --------------------------
def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


# --------------------------
# Strict schema check
# default_config defines the schema
# paper_config must have identical structure
# --------------------------
def check_schema(default_section, paper_section, prefix=""):
    for key in default_section:
        if key not in paper_section:
            continue

        # nested dict recursion
        if isinstance(default_section[key], dict):
            if not isinstance(paper_section[key], dict):
                raise ValueError(
                    f"Key '{prefix}{key}' must be an object/dict in paper_config"
                )
            check_schema(default_section[key], paper_section[key], prefix + key + ".")


# --------------------------
# Merge: paper overrides default
# If key exists in paper but not in default, add it
# --------------------------
def merge_dict(default, override):
    result = {}
    # First, process all keys from default
    for key, val in default.items():
        if key not in override:
            result[key] = val
        elif isinstance(val, dict):
            result[key] = merge_dict(val, override[key])
        else:
            result[key] = override[key]

    # Then, add any keys from override that are not in default
    for key, val in override.items():
        if key not in default:
            result[key] = val

    return result


# --------------------------
# FinalConfig for train.py
# --------------------------
class FinalConfig:
    def __init__(self):
        default_cfg = load_json("config_default.json")
        paper_cfg = load_json("config_paper.json")

        # 1. Verify top-level fields
        for section in ["model", "train", "data"]:
            if section not in paper_cfg:
                raise ValueError(f"Missing top-level section in paper_config: {section}")

        # 2. Strict schema validation (structure must match)
        for section in ["model", "train", "data"]:
            check_schema(default_cfg[section], paper_cfg[section], prefix=section + ".")

        # 3. Merge (paper overrides default)
        final_cfg = {}
        for section in ["model", "train", "data"]:
            final_cfg[section] = merge_dict(default_cfg[section], paper_cfg[section])

        # 4. Expose to train.py
        self.model = final_cfg["model"]
        self.train = final_cfg["train"]
        self.data = final_cfg["data"]

    def as_dict(self):
        return {
            "model": self.model,
            "train": self.train,
            "data": self.data
        }
