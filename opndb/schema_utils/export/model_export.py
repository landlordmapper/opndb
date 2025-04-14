from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Final, Any
import inspect

from pandera.api.base.model import MetaModel as PanderaMetaModel
from pandera import Column as PanderaColumn, DataFrameSchema

from opndb.validator.df_model import OPNDFModel
import importlib

BASE_PATH: Final[Path] = Path(__file__)
SCHEMA_MODULE_NAME: Final[str] = "schema_raw" # todo: clarify how we refer to models


def normalize_schema_string(schema: str) -> str:
    return schema.lower().strip().replace(".", "_")


@dataclass
class ExportableSchemaColumn:
    title: str
    nullable: bool
    description: str | None = None
    type: str | None = None
    default: str | None = None

    @classmethod
    def from_column(cls, column: PanderaColumn) -> "ExportableSchemaColumn":
        return cls(
            title=column.name,
            type=str(column.dtype),
            description=column.description,
            default=str(column.default) if column.default else None,
            nullable=column.nullable,
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
            columns=[
                ExportableSchemaColumn.from_column(c) for c in schema.columns.values()
            ],
        )


@dataclass
class ExportReport:
    """
    Schema report that can be exported to multiple text/file formats
    """

    version_code: str
    schemas: list[ExportableSchema]

    def to_md(self, file_path: str | None = None) -> str:
        """
        Convert the report to a markdown, returning the markdown.

        If passing file_path, it saves it to the file and returns the file path.

        todo: should the return change based on params? This doesn't seem right

        Returns:
            String markdown representation
        """
        lines = [f"## opndb schema version **{self.version_code}**\n"]
        for schema in self.schemas:
            lines.append(f"### {schema.name}\n")
            lines.append(f"{schema.description if schema.description else 'No Description'}\n")
            for col in schema.columns:
                default_str = f" = {col.default}" if col.default else ""
                lines.append(f"- {col.title}: {col.type}{default_str} {'(Required)' if col.nullable else ''}")
            lines.append("")  # Blank line between schemas
        out = "\n".join(lines)
        if file_path:
            with open(file_path, "w") as f:
                f.write(out)
                return file_path
        return out


    def to_pdf(self, file_path: str) -> str:
        """
        Convert the report to a PDF

        Returns:
            Path to the PDF file
        """
        from weasyprint import HTML
        from markdown.core import markdown
        html_text = markdown(self.to_md())
        HTML(string=html_text).write_pdf(file_path)
        return file_path




def create_report_for_schema(schema_name: str) -> ExportReport:
    schemas: list[OPNDFModel] = exportables_by_schema_name(schema_name)
    return ExportReport(
        version_code=normalize_schema_string(schema_name),
        schemas=[ExportableSchema.from_meta_model(s) for s in schemas],
    )


def get_yaml_for_schema(schema_name: str):
    schemas: list[OPNDFModel] = exportables_by_schema_name(schema_name)
    s = ""
    for schema in schemas:
        s += "\n---\n" + schema.to_yaml() + "\n"
    return s


def exportables_by_schema_name(schema_name: str) -> list[OPNDFModel]:
    normalized_schema_name: str = normalize_schema_string(schema_name)
    module_name = f"opndb.schema.{normalized_schema_name}.{SCHEMA_MODULE_NAME}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(f"Schema {normalized_schema_name} not found in module {module_name}") from e
    return get_exportables_from_module(module)


def get_exportables_from_module(module: ModuleType) -> list[OPNDFModel]:
    """
    Get the things we can export from the models in the module.

    They need to be of type OPNDFModel.
    """
    from opndb.validator.df_model import OPNDFModel

    members: list[tuple[str, Any]] = inspect.getmembers(module, inspect.isclass)
    exportables = [
        cls
        for _, cls in members
        if issubclass(cls, OPNDFModel) and cls is not OPNDFModel
    ]
    return exportables
