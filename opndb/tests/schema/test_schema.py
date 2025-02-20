from typing import Final
import pytest

from opndb.schema_utils.export.model_export import create_report_for_schema

EXAMPLE_SCHEMA: Final[str] = "v0_1"

VALID_SCHEMA__VALID_SCHEMA_STRINGS: Final[dict[str, list[str]]]  = {"v0_1": ["V0_1", "v0.1"]}

class TestSchema:
    class TestCreateReport:

        def test_basic(self):
            """
            basic test to ensure that a report can be made
            """
            report = create_report_for_schema(EXAMPLE_SCHEMA)
            assert report
            assert report.version_code == EXAMPLE_SCHEMA

        @pytest.mark.parametrize(
            "schema,values",
            [(schema, values) for schema, values in VALID_SCHEMA__VALID_SCHEMA_STRINGS.items()]
        )
        def test_alternatives_resolve_to_schema(self, schema: str, values: str):
            for v in values:
                assert schema == create_report_for_schema(v).version_code
            assert schema == create_report_for_schema(schema).version_code

