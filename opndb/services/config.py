import json
from pathlib import Path
from typing import Any, IO

from opndb.types.base import WorkflowConfigs


class ConfigManager:

    def __init__(self, config_path: Path | None = None):
        self._config_path = config_path or Path(__file__).parent.parent / "config.json"
        self._config: dict[str, Any] = {}
        self.load()

    def generate(self, root: str) -> None:
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
                "root": Path(root),
            }
            json_configs = {
                "root": str(configs["root"].absolute())
            }
            # Write configs to file with pretty printing
            with open(self._config_path, 'w', encoding='utf-8') as f:
                json.dump(json_configs, f, indent=2)
            # Update internal config state
            self._config = configs

        except OSError as e:
            raise OSError(f"Failed to create config file: {e}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to encode config data: {e.msg}", e.doc, e.pos)
        except Exception as e:
            raise Exception(f"Unexpected error generating config file: {e}")

    def load(self) -> None:
        """Load configuration from file"""
        if self._config_path.exists():
            with open(self._config_path) as f:
                self._config = json.load(f)

    def save(self) -> None:
        """Save current configuration to file"""
        with open(str(self._config_path), "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self._config[key] = value
        self.save()

    @property
    def exists(self) -> bool:
        return self._config_path.exists()

    @property
    def config(self) -> dict[str, Any]:
        """Public read-only access to configuration"""
        return self._config.copy()