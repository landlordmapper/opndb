from typing import Any


class Raw:
    """Column name constants for datasets stored in 'raw' data directory."""
    class TaxpayerRecords:

        PIN: str = "pin"
        TAX_NAME: str = "tax_name"
        TAX_ADDR: str = "tax_addr_full"
        TAX_STREET: str = "tax_street"
        TAX_CITY: str = "tax_city"
        TAX_STATE: str = "tax_state"
        TAX_ZIP: str = "tax_zip"
        BLDG_CLASS: str = "bldg_class"

        DTYPES: dict[str, Any] = {
            PIN: "str",
            TAX_NAME: "str",
            TAX_ADDR: "str",
            TAX_STREET: "str",
            TAX_CITY: "str",
            TAX_STATE: "str",
            TAX_ZIP: "str",
            BLDG_CLASS: "str",
        }

    class Corps:
        NAME: str = "name"
        FILE_NUMBER: str = "file_number"
        DATE_INCORPORATED: str = "date_incorporated"
        PRESIDENT_NAME: str = "president_name"
        PRESIDENT_ADDR: str = "president_addr"
        PRESIDENT_ADDR_STREET: str = "president_addr_street"
        PRESIDENT_ADDR_CITY: str = "president_addr_city"
        PRESIDENT_ADDR_STATE: str = "president_addr_state"
        PRESIDENT_ADDR_ZIP: str = "president_addr_zip"
        SECRETARY_NAME: str = "secretary_name"
        SECRETARY_ADDR: str = "secretary_addr"
        SECRETARY_ADDR_STREET: str = "secretary_addr_street"
        SECRETARY_ADDR_CITY: str = "secretary_addr_city"
        SECRETARY_ADDR_STATE: str = "secretary_addr_state"
        SECRETARY_ADDR_ZIP: str = "secretary_addr_zip"

    class LLCs:
        NAME: str = "name"
        FILE_NUMBER: str = "file_number"
        DATE_INCORPORATED: str = "date_incorporated"
        MANAGER_MEMBER_NAME: str = "manager_member_name"
        MANAGER_MEMBER_ADDR: str = "manager_member_addr"
        MANAGER_MEMBER_ADDR_STREET: str = "manager_member_addr"
        MANAGER_MEMBER_ADDR_CITY: str = "manager_member_addr"
        MANAGER_MEMBER_ADDR_STATE: str = "manager_member_addr"
        MANAGER_MEMBER_ADDR_ZIP: str = "manager_member_addr"
        AGENT_NAME: str = "agent_name"
        AGENT_ADDR: str = "agent_addr"
        AGENT_ADDR_STREET: str = "agent_addr_street"
        AGENT_ADDR_CITY: str = "agent_addr_city"
        AGENT_ADDR_STATE: str = "agent_addr_state"
        AGENT_ADDR_ZIP: str = "agent_addr_zip"
        OFFICE_ADDR: str = "office_addr"
        OFFICE_ADDR_STREET: str = "office_addr_street"
        OFFICE_ADDR_CITY: str = "office_addr_city"
        OFFICE_ADDR_STATE: str = "office_addr_state"
        OFFICE_ADDR_ZIP: str = "office_addr_zip"


    class ClassCodeDescriptions:
        CODE: str = "code"
        CATEGORY: str = "category"
        DESCRIPTION: str = "description"
        IS_RENTAL: str = "is_rental"


class Processed:
    """Column name constants for datasets stored in 'processed' data directory."""
    class ValidatedAddrs:
        pass

    class UnvalidatedAddrs:
        FULL_ADDRESS: str = "full_address"
        STREET: str = "street"
        CITY: str = "city"
        STATE: str = "state"
        ZIP: str = "zip"