import json
from pathlib import Path
from typing import Any, IO

from opndb.types.base import WorkflowConfigs
from opndb.utils import console


class ConfigManager:

    def __init__(self, configs_path: Path | None = None):
        self._configs_path = configs_path or Path(__file__).parent.parent / "configs.json"
        self._configs: dict[str, Any] = {}
        self.load()

    def generate(self, data_root: Path) -> None:
        """
        Generates new configs.json file.

        Args:
            root: Root directory path for the workflow
        Raises:
            OSError: If there are permission issues or path creation fails
            json.JSONDecodeError: If there are issues encoding the config to JSON
        """
        try:
            # Create config dictionary with typed structure
            configs: WorkflowConfigs = {
                "data_root": data_root,
                "load_ext": "csv"  # todo: remove, have user indicate (per-file?)
            }
            json_configs = {
                "data_root": str(configs["data_root"].absolute()),
                "load_ext": ".csv"
            }
            # Write configs to file with pretty printing
            with open(self._configs_path, "w", encoding="utf-8") as f:
                json.dump(json_configs, f, indent=2)
            # Update internal config state
            self._configs = configs
        except OSError as e:
            raise OSError(f"Failed to create config file: {e}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to encode config data: {e.msg}", e.doc, e.pos)
        except Exception as e:
            raise Exception(f"Unexpected error generating config file: {e}")

    def load(self) -> None:
        """Load configuration from file"""
        if self._configs_path.exists():
            with open(self._configs_path) as f:
                self._configs = json.load(f)

    def save(self) -> None:
        """Save current configuration to file"""
        with open(str(self._configs_path), "w", encoding="utf-8") as f:
            json.dump(self._configs, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._configs.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self._configs[key] = value
        self.save()

    @property
    def path(self) -> str:
        return str(self._configs_path)

    @property
    def exists(self) -> bool:
        found = self._configs_path.exists()
        if found:
            return found
        else:
            console.print("configs file not found")
        # return self._configs_path.exists()

    @property
    def configs(self) -> dict[str, Any]:
        """Public read-only access to configuration"""
        return self._configs.copy()