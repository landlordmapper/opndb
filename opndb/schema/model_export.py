from pathlib import Path
from types import ModuleType
from typing import Final, Any
import inspect

import opndb.schema.v0_1
from opndb.validator.df_model import OPNDFModel
import importlib
import os

BASE_PATH: Final[Path] = Path(__file__)
SCHEMA_FILE_NAME: Final[str] = "schema.py"

def get_yaml_for_schema(schema_name: str, *, stream: os.PathLike | None = None):
    schemas: list[OPNDFModel] = exportables_by_schema_name(schema_name)
    s = ""
    for schema in schemas:
        s += "\n---\n"+schema.to_yaml()+"\n"
    return s


def exportables_by_schema_name(schema_name: str) -> list[OPNDFModel]:
    module_name = f"opndb.schema.{schema_name}.schema"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Schema {schema_name} not found") from e
    return get_exportables_from_module(module)



def get_exportables_from_module(module: ModuleType) -> list[OPNDFModel]:
    from opndb.validator.df_model import OPNDFModel
    members: list[tuple[str, Any]] = inspect.getmembers(module, inspect.isclass)
    exportables = [
        cls for _, cls in members
        if issubclass(cls, OPNDFModel) and cls is not OPNDFModel
    ]
    return exportables
