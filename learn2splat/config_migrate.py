"""Config-version check for checkpoint configs.

Saved configs carry a `version`. `migrate` update a loaded config into `CURRENT_CFG_VERSION`. 
"""

CURRENT_CFG_VERSION = 2.5


def migrate(cfg_dict):
    # A missing key means a config written before versioning; null means a fresh run from main.yaml.
    version = cfg_dict.get("version", 0)
    if version is None:
        version = CURRENT_CFG_VERSION

    # Upgrade steps for future schema changes go here, e.g.
    #   if version == 2.5:
    #       cfg_dict = migrate_v2_5_to_v2_6(cfg_dict)
    #       version = 2.6

    if version != CURRENT_CFG_VERSION:
        raise ValueError(
            f"Unsupported config version {version}: this code loads checkpoint configs of "
            f"version {CURRENT_CFG_VERSION}.")
    return cfg_dict