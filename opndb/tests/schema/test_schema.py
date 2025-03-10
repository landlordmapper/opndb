import tempfile
from pathlib import Path

import markdown
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

    class TestToMd:

        def test_basic(self):
            """
            basic test to ensure that a report can be made
            """
            report = create_report_for_schema(EXAMPLE_SCHEMA)
            report_md = report.to_md()
            assert markdown.markdown(report_md), "Markdown should be returned"


    class TestToPdf:
        def test_basic(self):
            report = create_report_for_schema(EXAMPLE_SCHEMA)
            with tempfile.TemporaryDirectory() as temp_dir:
                path = report.to_pdf(temp_dir + "/foo.pdf")
                assert Path(path).exists()


