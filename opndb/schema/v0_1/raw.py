from typing import Any
import pandera as pa
from opndb.validator.df_model import OPNDFModel


class PropsTaxpayers(OPNDFModel):
    """
    Raw dataset containing both property and taxpayer record data. The opndb workflow will split up this dataset into
    separate datasets: one for taxpayer records, the other for properties.

    Constants defined at top of class are used for data processing, specifically to indicate which columns should be
    renamed, iterated over, etc. Each constant corresponds to a class method returning the value of the constant.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _RAW: list[str] = [
        "tax_name",
        "tax_street",
        "tax_city",
        "tax_state",
        "tax_zip",
    ]
    _CLEAN_RENAME_MAP: dict[str, Any] = {
        "tax_name": "clean_name",
        "tax_street": "clean_street",
        "tax_city": "clean_city",
        "tax_state": "clean_state",
        "tax_zip": "clean_zip",
    }
    _RAW_ADDRESS_MAP: list[dict[str, Any]] = [
        {
            "full_address": "raw_address",
            "address_cols": [
                "raw_street",
                "raw_city",
                "raw_state",
                "raw_zip",
            ]
        }
    ]
    _NAME_ADDRESS_CONCAT_MAP: dict[str, Any] = {
        "raw": {
            "name_addr": "raw_name_address",
            "name": "raw_name",
            "addr": "raw_address"
        },
        "clean": {
            "name_addr": "clean_name_address",
            "name": "clean_name",
            "addr": "clean_address"
        }
    }
    _BASIC_CLEAN: list[str] = [
        "pin",
        "class_code",
        "clean_name",
        "clean_street",
        "clean_city",
        "clean_state",
        "clean_zip",
    ]
    _NAME_CLEAN: list[str] = ["clean_name"]
    _ADDRESS_CLEAN: dict[str, Any] = {
        "street": ["clean_street"],
        "zip": ["clean_zip"],
    }
    _CLEAN_ADDRESS_MAP: list[dict[str, Any]] = [
        {
            "full_address": "clean_address",
            "address_cols": [
                "clean_street",
                "clean_city",
                "clean_state",
                "clean_zip",
            ]
        }
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, Any]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def raw_address_map(cls) -> list[dict[str, Any]]:
        return cls._RAW_ADDRESS_MAP

    @classmethod
    def name_address_concat_map(cls) -> dict[str, Any]:
        return cls._NAME_ADDRESS_CONCAT_MAP

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    @classmethod
    def name_clean(cls) -> list[str]:
        return cls._NAME_CLEAN

    @classmethod
    def address_clean(cls) -> dict[str, Any]:
        return cls._ADDRESS_CLEAN

    @classmethod
    def clean_address_map(cls) -> list[dict[str, Any]]:
        return cls._CLEAN_ADDRESS_MAP

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    pin: str = pa.Field(
        nullable=False,
        unique=True,
        title="PIN",
        description="Unique tax identifier for the property",
    )
    tax_name: str = pa.Field(
        nullable=False,
        title="Taxpayer Name",
        description="Taxpayer name indicated for the property",
    )
    tax_street: str = pa.Field(
        nullable=False,
        title="Taxpayer Street",
        description="Street address for the taxpayer of the property, including street number, street name, secondary/unit number and all prefixes and suffixes.",
    )
    tax_city: str = pa.Field(
        nullable=False,
        title="Taxpayer City",
        description="City associated with property taxpayer's street address."
    )
    tax_state: str = pa.Field(
        nullable=False,
        title="Taxpayer State",
        description="State associated with property taxpayer's street address."
    )
    tax_zip: str = pa.Field(
        nullable=False,
        title="Taxpayer Zip",
        description="Zip code associated with property taxpayer's street address."
    )
    class_code: str = pa.Field(
        nullable=False,
        title="Class Code",
        description="Municipal code indicating land use for the property, required for subsetting rental properties."
    )
    num_units: int | None = pa.Field(
        nullable=True,
        title="Number of Units",
        description="Number of rental apartment units in the property."
    )

class Corps(OPNDFModel):
    """
    Raw dataset for state-level corporate records. Note that the availability of address-related columns is subject to
    the quality of the original data and the ability to parse complete address strings.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _RAW: list[str] = [
        "name",
        "president_name",
        "president_address",
        "secretary_name",
        "secretary_address",
    ]
    _CLEAN_RENAME_MAP: dict[str, str] = {
        "name": "clean_name",
        "president_name": "clean_president_name",
        "president_address": "clean_president_address",
        "secretary_name": "clean_secretary_name",
        "secretary_address": "clean_secretary_address",
    }
    _RAW_ADDRESS_MAP: None = None
    _BASIC_CLEAN: list[str] = [
        "file_number",
        "status",
        "clean_name",
        "clean_president_name",
        "clean_president_address",
        "clean_secretary_name",
        "clean_secretary_address",
    ]
    _NAME_CLEAN: list[str] = [
        "clean_name",
        "clean_president_name",
        "clean_secretary_name",
    ]
    _ADDRESS_CLEAN: dict[str, list[str]] = {
        "street": [
            "clean_president_address",
            "clean_secretary_address",
        ],
        "zip": []
    }
    _CLEAN_ADDRESS_MAP: None = None
    _OUT: list[str] = [
        "file_number",
        "status",
        "raw_name",
        "raw_president_name",
        "raw_president_address",
        "raw_secretary_name",
        "raw_secretary_address",
        "clean_name",
        "clean_president_name",
        "clean_president_address",
        "clean_secretary_name",
        "clean_secretary_address",
    ]
    _UNVALIDATED_COL_OBJS: list[dict[str, str]] = [
        {
            "raw_president_address": "raw_address",
            "clean_president_address": "clean_address",
            "status": "status",
        },
        {
            "raw_secretary_address": "raw_address",
            "clean_secretary_address": "clean_address",
            "status": "status",
        }
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, str]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def raw_address_map(cls) -> None:
        return cls._RAW_ADDRESS_MAP

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    @classmethod
    def name_clean(cls) -> list[str]:
        return cls._NAME_CLEAN

    @classmethod
    def address_clean(cls) -> dict[str, list[str]]:
        return cls._ADDRESS_CLEAN

    @classmethod
    def clean_address_map(cls) -> None:
        return cls._CLEAN_ADDRESS_MAP

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

    @classmethod
    def unvalidated_col_objs(cls) -> list[dict[str, str]]:
        return cls._UNVALIDATED_COL_OBJS

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    name: str = pa.Field(
        nullable=False,
        unique=True,
        title="Corporation Name",
        description="Corporation name",
    )
    file_number: str = pa.Field(
        nullable=False,
        unique=True,
        title="Corporation File Number",
        description="Unique identifier number assigned to corporation by secretary of state upon incorporation.",
    )
    date_incorporated: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Incorporated",
        description="Date Incorporated",
    )
    date_dissolved: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Dissolved",
        description="Date Dissolved",
    )
    status: str | None = pa.Field(
        nullable=False,
        unique=False,
        title="Status",
        description="Organization's status (active, inactive, involuntarily dissolved, etc.)",
    )
    president_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="President Name",
    )
    president_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="President Address",
        description="Complete concatenated mailing address for corporation president",
    )
    president_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President Street",
        description="President street",
    )
    president_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President City",
        description="President city",
    )
    president_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President State",
        description="President state",
    )
    president_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="President Zip Code",
        description="President zip code",
    )
    secretary_name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Name",
        description="Secretary name",
    )
    secretary_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Address",
        description="Secretary address",
    )
    secretary_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Street",
        description="Secretary street",
    )
    secretary_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary City",
        description="Secretary city",
    )
    secretary_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary State",
        description="Secretary state",
    )
    secretary_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Zip Code",
        description="Secretary zip code",
    )

class LLCs(OPNDFModel):
    """
    Raw dataset for state-level LLC records. Note that the availability of address-related columns is subject to
    the quality of the original data and the ability to parse complete address strings.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _RAW: list[str] = [
        "name",
        "manager_member_name",
        "manager_member_street",
        "manager_member_city",
        "manager_member_zip",
        "agent_name",
        "agent_street",
        "agent_zip",
        "office_street",
        "office_city",
        "office_zip",
    ]
    _CLEAN_RENAME_MAP: dict[str, str] = {
        "name": "clean_name",
        "manager_member_name": "clean_manager_member_name",
        "manager_member_street": "clean_manager_member_street",
        "manager_member_city": "clean_manager_member_city",
        "manager_member_zip": "clean_manager_member_zip",
        "agent_name": "clean_agent_name",
        "agent_street": "clean_agent_street",
        "agent_zip": "clean_agent_zip",
        "office_street": "clean_office_street",
        "office_city": "clean_office_city",
        "office_zip": "clean_office_zip",
    }
    _RAW_ADDRESS_MAP: list[dict[str, list[str]]] = [
        {
            "full_address": "raw_manager_member_address",
            "address_cols": [
                "raw_manager_member_street",
                "raw_manager_member_city",
                "raw_manager_member_zip",
            ],
        },
        {
            "full_address": "raw_agent_address",
            "address_cols": [
                "raw_agent_street",
                "raw_agent_zip",
            ],
        },
        {
            "full_address": "raw_office_address",
            "address_cols": [
                "raw_office_street",
                "raw_office_city",
                "raw_office_zip",
            ],
        },
    ]
    _BASIC_CLEAN: list[str] = [
        "file_number",
        "status",
        "clean_name",
        "clean_manager_member_name",
        "clean_manager_member_street",
        "clean_manager_member_city",
        "clean_manager_member_zip",
        "clean_agent_name",
        "clean_agent_street",
        "clean_agent_zip",
        "clean_office_street",
        "clean_office_city",
        "clean_office_zip",
    ]
    _NAME_CLEAN: list[str] = [
        "clean_name",
        "clean_manager_member_name",
        "clean_agent_name",
    ]
    _ADDRESS_CLEAN: dict[str, list[str]] = {
        "street": [
            "clean_manager_member_street",
            "clean_agent_street",
            "clean_office_street",
        ],
        "zip": [
            "clean_manager_member_zip",
            "clean_agent_zip",
            "clean_office_zip",
        ],
    }
    _CLEAN_ADDRESS_MAP: list[dict[str, list[str]]] = [
        {
            "full_address": "clean_manager_member_address",
            "address_cols": [
                "clean_manager_member_street",
                "clean_manager_member_city",
                "clean_manager_member_zip",
            ],
        },
        {
            "full_address": "clean_agent_address",
            "address_cols": [
                "clean_agent_street",
                "clean_agent_zip",
            ],
        },
        {
            "full_address": "clean_office_address",
            "address_cols": [
                "clean_office_street",
                "clean_office_city",
                "clean_office_zip",
            ],
        },
    ]
    _OUT: list[str] = [
        "file_number",
        "status",
        "raw_name",
        "raw_manager_member_name",
        "raw_manager_member_street",
        "raw_manager_member_city",
        "raw_manager_member_zip",
        "raw_manager_member_address",
        "raw_agent_name",
        "raw_agent_street",
        "raw_agent_zip",
        "raw_agent_address",
        "raw_office_street",
        "raw_office_city",
        "raw_office_zip",
        "raw_office_address",
        "clean_name",
        "clean_manager_member_name",
        "clean_manager_member_street",
        "clean_manager_member_city",
        "clean_manager_member_zip",
        "clean_manager_member_address",
        "clean_agent_name",
        "clean_agent_street",
        "clean_agent_zip",
        "clean_agent_address",
        "clean_office_street",
        "clean_office_city",
        "clean_office_zip",
        "clean_office_address",
    ]
    _UNVALIDATED_COL_OBJS: list[dict[str, str]] = [
        {
            "raw_manager_member_street": "raw_street",
            "raw_manager_member_city": "raw_city",
            "raw_manager_member_zip": "raw_zip",
            "raw_manager_member_address": "raw_address",
            "clean_manager_member_street": "clean_street",
            "clean_manager_member_city": "clean_city",
            "clean_manager_member_zip": "clean_zip",
            "clean_manager_member_address": "clean_address",
            "status": "status",
        },
        {
            "raw_agent_street": "raw_street",
            "raw_agent_zip": "raw_zip",
            "raw_agent_address": "raw_address",
            "clean_agent_street": "clean_street",
            "clean_agent_zip": "clean_zip",
            "clean_agent_address": "clean_address",
            "status": "status",
        },
        {
            "raw_office_street": "raw_street",
            "raw_office_city": "raw_city",
            "raw_office_zip": "raw_zip",
            "raw_office_address": "raw_address",
            "clean_office_street": "clean_street",
            "clean_office_city": "clean_city",
            "clean_office_zip": "clean_zip",
            "clean_office_address": "clean_address",
            "status": "status",
        },
    ]

    @classmethod
    def raw(cls) -> list[str]:
        return cls._RAW

    @classmethod
    def clean_rename_map(cls) -> dict[str, str]:
        return cls._CLEAN_RENAME_MAP

    @classmethod
    def raw_address_map(cls) -> list[dict[str, list[str]]]:
        return cls._RAW_ADDRESS_MAP

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    @classmethod
    def name_clean(cls) -> list[str]:
        return cls._NAME_CLEAN

    @classmethod
    def address_clean(cls) -> dict[str, list[str]]:
        return cls._ADDRESS_CLEAN

    @classmethod
    def clean_address_map(cls) -> list[dict[str, list[str]]]:
        return cls._CLEAN_ADDRESS_MAP

    @classmethod
    def out(cls) -> list[str]:
        return cls._OUT

    @classmethod
    def unvalidated_col_objs(cls) -> list[dict[str, str]]:
        return cls._UNVALIDATED_COL_OBJS

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    name: str = pa.Field(
        nullable=False,
        unique=True,
        title="LLC Name",
        description="LLC name",
    )
    file_number: str = pa.Field(
        nullable=False,
        unique=True,
        title="LLC File Number",
        description="Unique identifier number assigned to LLC by secretary of state upon incorporation.",
    )
    date_incorporated: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Incorporated",
        description="Date Incorporated",
    )
    date_dissolved: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Date Dissolved",
        description="Date Dissolved",
    )
    status: str = pa.Field(
        nullable=False,
        unique=False,
        title="Status",
        description="Organization's status (active, inactive, involuntarily dissolved, etc.)",
    )
    manager_member_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Manager/Member Name",
        description="Manager/member name",
    )
    manager_member_address: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Address",
        description="Manager/member address",
    )
    manager_member_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Street",
        description="Manager/member street",
    )
    manager_member_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member City",
        description="Manager/member city",
    )
    manager_member_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member State",
        description="Manager/member state",
    )
    manager_member_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Zip Code",
        description="Manager/member zip code",
    )
    agent_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Agent Name",
        description="Agent name",
    )
    agent_address: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Address",
        description="Agent address",
    )
    agent_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Street",
        description="Agent street",
    )
    agent_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent City",
        description="Agent city",
    )
    agent_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent State",
        description="Agent state",
    )
    agent_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Zip Code",
        description="Agent zip code",
    )
    office_address: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office Address",
        description="Office address",
    )
    office_street: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office Street",
        description="Office street",
    )
    office_city: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office City",
        description="Office city",
    )
    office_state: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office State",
        description="Office state",
    )
    office_zip: str | None = pa.Field(
        nullable=True,
        unique=False,
        title="Office Zip Code",
        description="Office zip code",
    )

class ClassCodes(OPNDFModel):
    """
    Dataset containing building class codes and their meaning. Usually set by the municipality to dictate zoning. Used
    to subset rental properties.
    """
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _BASIC_CLEAN: list[str] = [
        "code",
        "category",
        "description"
    ]

    @classmethod
    def basic_clean(cls) -> list[str]:
        return cls._BASIC_CLEAN

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    code: str = pa.Field(
        nullable=False,
        unique=True,
        title="Class Code",
        description="Code provided by municipal or county government designating land use. Used to subset rental properties.",
    )
    category: str = pa.Field(
        nullable=True,
        unique=False,
        title="Class Category",
        description="Optional categorization/descriptor associated with class code.",
    )
    description: str = pa.Field(
        nullable=True,
        title="Class Description",
        description="Detailed description defining land use for class code.",
    )
    is_rental: bool = pa.Field(
        nullable=False,
        title="Is Rental?",
        description="Boolean column indicating whether or not the class code is associated with rental properties."
    )
