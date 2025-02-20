from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Any
import inspect

from pandera.api.base.model import MetaModel as PanderaMetaModel
from pandera import Column as PanderaColumn, DataFrameSchema

import opndb.schema.v0_1
from opndb.validator.df_model import OPNDFModel
import importlib
import os

BASE_PATH: Final[Path] = Path(__file__)
SCHEMA_FILE_NAME: Final[str] = "schema.py"

@dataclass
class ExportableSchemaColumn:
    title: str
    description: str | None = None
    type: str | None = None
    default: str  | None = None

    @classmethod
    def from_column(cls, column: PanderaColumn) -> "ExportableSchemaColumn":
        return cls(
            title=column.name,
            type=str(column.dtype),
            description=column.description,
            default=str(column.default) if column.default else None,
        )

@dataclass
class ExportableSchema:
    name: str
    description: str
    columns: list[ExportableSchemaColumn]

    @classmethod
    def from_meta_model(cls, model: PanderaMetaModel) -> "ExportableSchema":
        schema: DataFrameSchema = model.to_schema()

        return cls(
            name=schema.name,
            description=schema.description,
            columns=[ExportableSchemaColumn.from_column(c) for c in schema.columns.values()]
        )


@dataclass
class ExportReport:
    version_code: str
    schemas: list[ExportableSchema]

    def to_pdf(self, output_path: Path):
        pass


def create_report_for_schema(schema_name: str) -> ExportReport:
    schemas: list[OPNDFModel] = exportables_by_schema_name(schema_name)
    return ExportReport(
        version_code=schema_name,
        schemas=[ExportableSchema.from_meta_model(s) for s in schemas]
    )

def get_yaml_for_schema(schema_name: str):
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
