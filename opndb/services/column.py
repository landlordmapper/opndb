from pydantic import BaseModel

class ColumnManagerBase(BaseModel):
    pass


class ColumnClassCodes(ColumnManagerBase):

    CODE: str = "code"
    CATEGORY: str = "category"
    DESCRIPTION: str = "description"
    IS_RENTAL: str = "is_rental"

    @property
    def basic_clean(self):
        return [
            self.CODE,
            self.CATEGORY,
            self.DESCRIPTION,
            self.IS_RENTAL,
        ]


class ColumnPropsTaxpayers(ColumnManagerBase):

    # raw data columns
    PIN: str = "pin"
    TAX_NAME: str = "tax_name"
    TAX_ADDRESS: str = "tax_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    TAX_STREET: str = "tax_street"  # todo: add documentation: optional - addresses may come in a single string format and need to be parsed
    TAX_CITY: str = "tax_city"
    TAX_STATE: str = "tax_state"
    TAX_ZIP: str = "tax_zip"
    CLASS_CODE: str = "class_code"
    NUM_UNITS: str = "num_units"  # todo: add documentation: optional - ideal but will work without it

    # processing columns
    RAW_NAME: str = "raw_name"
    RAW_STREET: str = "raw_street"
    RAW_CITY: str = "raw_city"
    RAW_STATE: str = "raw_state"
    RAW_ZIP: str = "raw_zip"
    RAW_ADDRESS: str = "raw_address"
    RAW_NAME_ADDRESS: str = "raw_name_address"
    CLEAN_NAME: str = "clean_name"
    CLEAN_STREET: str = "clean_street"
    CLEAN_CITY: str = "clean_city"
    CLEAN_STATE: str = "clean_state"
    CLEAN_ZIP: str = "clean_zip"
    CLEAN_ADDRESS: str = "clean_address"
    CLEAN_NAME_ADDRESS: str = "clean_name_address"

    @property
    def raw(self):
        # exclude raw_address - this will be generated manually after creating the raw cols
        return [
            self.TAX_NAME,
            self.TAX_STREET,
            self.TAX_CITY,
            self.TAX_STATE,
            self.TAX_ZIP,
        ]

    @property
    def clean_rename_map(self):
        # exclude full address - this will be generated after cleaning
        return {
            self.TAX_NAME: self.CLEAN_NAME,
            self.TAX_STREET: self.CLEAN_STREET,
            self.TAX_CITY: self.CLEAN_CITY,
            self.TAX_STATE: self.CLEAN_STATE,
            self.TAX_ZIP: self.CLEAN_ZIP,
        }

    @property
    def raw_address_map(self):
        return [
            {
                "full_address": self.RAW_ADDRESS,
                "address_cols": [
                    self.RAW_STREET,
                    self.RAW_CITY,
                    self.RAW_STATE,
                    self.RAW_ZIP,
                ]
            }
        ]

    @property
    def name_address_concat_map(self):
        return {
            "raw": {
                "name_addr": self.RAW_NAME_ADDRESS,
                "name": self.RAW_NAME,
                "addr": self.RAW_ADDRESS,
            },
            "clean": {
                "name_addr": self.CLEAN_NAME_ADDRESS,
                "name": self.CLEAN_NAME,
                "addr": self.CLEAN_ADDRESS,
            }
        }

    @property
    def basic_clean(self):
        return [
            self.PIN,
            self.NUM_UNITS,
            self.CLASS_CODE,
            self.CLEAN_NAME,
            self.CLEAN_STREET,
            self.CLEAN_CITY,
            self.CLEAN_STATE,
            self.CLEAN_ZIP,
        ]

    @property
    def name_clean(self):
        return [
            self.CLEAN_NAME,
        ]

    @property
    def address_clean(self):
        return {
            "street": [
                self.CLEAN_STREET,
            ],
            "zip": [
                self.CLEAN_ZIP,
            ]
        }

    @property
    def clean_address_map(self):
        return [
            {
                "full_address": self.CLEAN_ADDRESS,
                "address_cols": [
                    self.CLEAN_STREET,
                    self.CLEAN_CITY,
                    self.CLEAN_STATE,
                    self.CLEAN_ZIP,
                ]
            }
        ]


class ColumnCorps(ColumnManagerBase):

    # raw data fields
    NAME: str = "name"
    FILE_NUMBER: str = "file_number"
    DATE_INCORPORATED: str = "date_incorporated"
    DATE_DISSOLVED: str = "date_dissolved"  # todo: add documentation: optional may be active
    STATUS: str = "status"

    PRESIDENT_NAME: str = "president_name"
    PRESIDENT_ADDRESS: str = "president_address"
    PRESIDENT_STREET: str = "president_street"
    PRESIDENT_CITY: str = "president_city"
    PRESIDENT_STATE: str = "president_state"
    PRESIDENT_ZIP: str = "president_zip"
    SECRETARY_NAME: str = "secretary_name"
    SECRETARY_ADDRESS: str = "secretary_address"
    SECRETARY_STREET: str = "secretary_street"
    SECRETARY_CITY: str = "secretary_city"
    SECRETARY_STATE: str = "secretary_state"
    SECRETARY_ZIP: str = "secretary_zip"

    # processed data fields
    RAW_NAME: str = "raw_name"
    RAW_PRESIDENT_NAME: str = "raw_president_name"
    RAW_PRESIDENT_ADDRESS: str = "raw_president_address"
    RAW_PRESIDENT_STREET: str = "raw_president_street"
    RAW_PRESIDENT_CITY: str = "raw_president_city"
    RAW_PRESIDENT_STATE: str = "raw_president_state"
    RAW_PRESIDENT_ZIP: str = "raw_president_zip"
    RAW_SECRETARY_NAME: str = "raw_secretary_name"
    RAW_SECRETARY_ADDRESS: str = "raw_secretary_address"
    RAW_SECRETARY_STREET: str = "raw_secretary_street"
    RAW_SECRETARY_CITY: str = "raw_secretary_city"
    RAW_SECRETARY_STATE: str = "raw_secretary_state"
    RAW_SECRETARY_ZIP: str = "raw_secretary_zip"

    CLEAN_NAME: str = "clean_name"
    CLEAN_PRESIDENT_NAME: str = "clean_president_name"
    CLEAN_PRESIDENT_ADDRESS: str = "clean_president_address"
    CLEAN_PRESIDENT_STREET: str = "clean_president_street"
    CLEAN_PRESIDENT_CITY: str = "clean_president_city"
    CLEAN_PRESIDENT_STATE: str = "clean_president_state"
    CLEAN_PRESIDENT_ZIP: str = "clean_president_zip"
    CLEAN_SECRETARY_NAME: str = "clean_secretary_name"
    CLEAN_SECRETARY_ADDRESS: str = "clean_secretary_address"
    CLEAN_SECRETARY_STREET: str = "clean_secretary_street"
    CLEAN_SECRETARY_CITY: str = "clean_secretary_city"
    CLEAN_SECRETARY_STATE: str = "clean_secretary_state"
    CLEAN_SECRETARY_ZIP: str = "clean_secretary_zip"

    CORE_NAME: str = "core_name"
    IS_BANK_PRESIDENT: str = "is_bank_president"
    IS_PERSON_PRESIDENT: str = "is_person_president"
    IS_COMMON_NAME_PRESIDENT: str = "is_common_name_president"
    IS_ORG_PRESIDENT: str = "is_org_president"
    IS_LLC_PRESIDENT: str = "is_llc_president"
    IS_BANK_SECRETARY: str = "is_bank_secretary"
    IS_PERSON_SECRETARY: str = "is_person_secretary"
    IS_COMMON_NAME_SECRETARY: str = "is_common_name_secretary"
    IS_ORG_SECRETARY: str = "is_org_secretary"
    IS_LLC_SECRETARY: str = "is_llc_secretary"

    @property
    def raw(self):
        return [
            self.NAME,
            self.PRESIDENT_NAME,
            self.PRESIDENT_ADDRESS,
            self.SECRETARY_NAME,
            self.SECRETARY_ADDRESS,
        ]

    @property
    def clean_rename_map(self):
        # exclude full address - this will be generated after cleaning
        return {
            self.NAME: self.CLEAN_NAME,
            self.PRESIDENT_NAME: self.CLEAN_PRESIDENT_NAME,
            self.PRESIDENT_ADDRESS: self.CLEAN_PRESIDENT_ADDRESS,
            self.SECRETARY_NAME: self.CLEAN_SECRETARY_NAME,
            self.SECRETARY_ADDRESS: self.CLEAN_SECRETARY_ADDRESS,
        }

    @property
    def raw_address_map(self):
        return None

    @property
    def basic_clean(self):
        return [
            self.FILE_NUMBER,
            self.STATUS,
            self.CLEAN_NAME,
            self.CLEAN_PRESIDENT_NAME,
            self.CLEAN_PRESIDENT_ADDRESS,
            self.CLEAN_SECRETARY_NAME,
            self.CLEAN_SECRETARY_ADDRESS,
        ]

    @property
    def name_clean(self):
        return [
            self.CLEAN_NAME,
            self.CLEAN_PRESIDENT_NAME,
            self.CLEAN_SECRETARY_NAME,
        ]

    @property
    def address_clean(self):
        return {
            "street": [
                self.CLEAN_PRESIDENT_ADDRESS,
                self.CLEAN_SECRETARY_ADDRESS,
            ],
            "zip": []
        }

    @property
    def clean_address_map(self):
        return None

    @property
    def out(self):
        return [
            self.FILE_NUMBER,
            self.STATUS,
            self.RAW_NAME,
            self.RAW_PRESIDENT_NAME,
            self.RAW_PRESIDENT_ADDRESS,
            self.RAW_SECRETARY_NAME,
            self.RAW_SECRETARY_ADDRESS,
            self.CLEAN_NAME,
            self.CLEAN_PRESIDENT_NAME,
            self.CLEAN_PRESIDENT_ADDRESS,
            self.CLEAN_SECRETARY_NAME,
            self.CLEAN_SECRETARY_ADDRESS,
        ]

    @property
    def unvalidated_col_objs(self):
        return [
            {
                self.RAW_PRESIDENT_ADDRESS: "raw_address",
                self.CLEAN_PRESIDENT_ADDRESS: "clean_address",
            },
            {
                self.RAW_SECRETARY_ADDRESS: "raw_address",
                self.CLEAN_SECRETARY_ADDRESS: "clean_address",
            }
        ]


class ColumnLLCs(ColumnManagerBase):

    # raw data fields
    NAME: str = "name"
    FILE_NUMBER: str = "file_number"
    DATE_INCORPORATED: str = "date_incorporated"
    DATE_DISSOLVED: str = "date_dissolved"  # todo: add documentation: optional may be active
    STATUS: str = "status"  # todo: add documentation: optional - may be codes whose meanings will determine how the active filters are applied

    MANAGER_MEMBER_NAME: str = "manager_member_name"
    MANAGER_MEMBER_ADDRESS: str = "manager_member_address"
    MANAGER_MEMBER_STREET: str = "manager_member_street"
    MANAGER_MEMBER_CITY: str = "manager_member_city"
    MANAGER_MEMBER_STATE: str = "manager_member_state"  # todo: add documentation: optional - sometimes missing, not ideal
    MANAGER_MEMBER_ZIP: str = "manager_member_zip"
    AGENT_NAME: str = "agent_name"
    AGENT_ADDRESS: str = "agent_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    AGENT_STREET: str = "agent_street"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    AGENT_CITY: str = "agent_city"
    AGENT_STATE: str = "agent_state"  # todo: add documentation: optional - sometimes missing, not ideal
    AGENT_ZIP: str = "agent_zip"
    OFFICE_ADDRESS: str = "office_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    OFFICE_STREET: str = "office_street"
    OFFICE_CITY: str = "office_city"
    OFFICE_STATE: str = "office_state"  # todo: add documentation: optional - sometimes missing, not ideal
    OFFICE_ZIP: str = "office_zip"

    RAW_NAME: str = "raw_name"
    RAW_MANAGER_MEMBER_NAME: str = "raw_manager_member_name"
    RAW_MANAGER_MEMBER_ADDRESS: str = "raw_manager_member_address"
    RAW_MANAGER_MEMBER_STREET: str = "raw_manager_member_street"
    RAW_MANAGER_MEMBER_CITY: str = "raw_manager_member_city"
    RAW_MANAGER_MEMBER_STATE: str = "raw_manager_member_state"  # todo: add documentation: optional - sometimes missing, not ideal
    RAW_MANAGER_MEMBER_ZIP: str = "raw_manager_member_zip"
    RAW_AGENT_NAME: str = "raw_agent_name"
    RAW_AGENT_ADDRESS: str = "raw_agent_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    RAW_AGENT_STREET: str = "raw_agent_street"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    RAW_AGENT_CITY: str = "raw_agent_city"
    RAW_AGENT_STATE: str = "raw_agent_state"  # todo: add documentation: optional - sometimes missing, not ideal
    RAW_AGENT_ZIP: str = "raw_agent_zip"
    RAW_OFFICE_ADDRESS: str = "raw_office_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    RAW_OFFICE_STREET: str = "raw_office_street"
    RAW_OFFICE_CITY: str = "raw_office_city"
    RAW_OFFICE_STATE: str = "raw_office_state"  # todo: add documentation: optional - sometimes missing, not ideal
    RAW_OFFICE_ZIP: str = "raw_office_zip"

    CLEAN_NAME: str = "clean_name"
    CLEAN_MANAGER_MEMBER_NAME: str = "clean_manager_member_name"
    CLEAN_MANAGER_MEMBER_ADDRESS: str = "clean_manager_member_address"
    CLEAN_MANAGER_MEMBER_STREET: str = "clean_manager_member_street"
    CLEAN_MANAGER_MEMBER_CITY: str = "clean_manager_member_city"
    CLEAN_MANAGER_MEMBER_STATE: str = "clean_manager_member_state"  # todo: add documentation: optional - sometimes missing, not ideal
    CLEAN_MANAGER_MEMBER_ZIP: str = "clean_manager_member_zip"
    CLEAN_AGENT_NAME: str = "clean_agent_name"
    CLEAN_AGENT_ADDRESS: str = "clean_agent_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    CLEAN_AGENT_STREET: str = "clean_agent_street"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    CLEAN_AGENT_CITY: str = "clean_agent_city"
    CLEAN_AGENT_STATE: str = "clean_agent_state"  # todo: add documentation: optional - sometimes missing, not ideal
    CLEAN_AGENT_ZIP: str = "clean_agent_zip"
    CLEAN_OFFICE_ADDRESS: str = "clean_office_address"  # todo: add documentation: optional - addresses may come broken down already into street, city, state and zip
    CLEAN_OFFICE_STREET: str = "clean_office_street"
    CLEAN_OFFICE_CITY: str = "clean_office_city"
    CLEAN_OFFICE_STATE: str = "clean_office_state"  # todo: add documentation: optional - sometimes missing, not ideal
    CLEAN_OFFICE_ZIP: str = "clean_office_zip"

    CORE_NAME: str = "core_name"
    IS_BANK_MANAGER_MEMBER: str = "is_bank_manager_member"
    IS_PERSON_MANAGER_MEMBER: str = "is_person_manager_member"
    IS_COMMON_NAME_MANAGER_MEMBER: str = "is_common_name_manager_member"
    IS_ORG_MANAGER_MEMBER: str = "is_org_secretary"
    IS_LLC_MANAGER_MEMBER: str = "is_llc_secretary"
    IS_BANK_AGENT: str = "is_bank_agent"
    IS_PERSON_AGENT: str = "is_person_agent"
    IS_COMMON_NAME_AGENT: str = "is_common_name_agent"
    IS_ORG_AGENT: str = "is_org_agent"
    IS_LLC_AGENT: str = "is_llc_agent"

    @property
    def raw(self):
        return [
            self.NAME,
            self.MANAGER_MEMBER_NAME,
            self.MANAGER_MEMBER_STREET,
            self.MANAGER_MEMBER_CITY,
            self.MANAGER_MEMBER_ZIP,
            self.AGENT_NAME,
            self.AGENT_STREET,
            self.AGENT_ZIP,
            self.OFFICE_STREET,
            self.OFFICE_CITY,
            self.OFFICE_ZIP,
        ]

    @property
    def clean_rename_map(self):
        return {
            self.NAME: self.CLEAN_NAME,
            self.MANAGER_MEMBER_NAME: self.CLEAN_MANAGER_MEMBER_NAME,
            self.MANAGER_MEMBER_STREET: self.CLEAN_MANAGER_MEMBER_STREET,
            self.MANAGER_MEMBER_CITY: self.CLEAN_MANAGER_MEMBER_CITY,
            self.MANAGER_MEMBER_ZIP: self.CLEAN_MANAGER_MEMBER_ZIP,
            self.AGENT_NAME: self.CLEAN_AGENT_NAME,
            self.AGENT_STREET: self.CLEAN_AGENT_STREET,
            self.AGENT_ZIP: self.CLEAN_AGENT_ZIP,
            self.OFFICE_STREET: self.CLEAN_OFFICE_STREET,
            self.OFFICE_CITY: self.CLEAN_OFFICE_CITY,
            self.OFFICE_ZIP: self.CLEAN_OFFICE_ZIP,
        }

    @property
    def raw_address_map(self):
        return [
            {
                "full_address": self.RAW_MANAGER_MEMBER_ADDRESS,
                "address_cols": [
                    self.RAW_MANAGER_MEMBER_STREET,
                    self.RAW_MANAGER_MEMBER_CITY,
                    self.RAW_MANAGER_MEMBER_ZIP,
                ]
            },
            {
                "full_address": self.RAW_AGENT_ADDRESS,
                "address_cols": [
                    self.RAW_AGENT_STREET,
                    self.RAW_AGENT_ZIP,
                ]
            },
            {
                "full_address": self.RAW_OFFICE_ADDRESS,
                "address_cols": [
                    self.RAW_OFFICE_STREET,
                    self.RAW_OFFICE_CITY,
                    self.RAW_OFFICE_ZIP,
                ]
            },
        ]

    @property
    def basic_clean(self):
        return [
            self.FILE_NUMBER,
            self.STATUS,
            self.CLEAN_NAME,
            self.CLEAN_MANAGER_MEMBER_NAME,
            self.CLEAN_MANAGER_MEMBER_STREET,
            self.CLEAN_MANAGER_MEMBER_CITY,
            self.CLEAN_MANAGER_MEMBER_ZIP,
            self.CLEAN_AGENT_NAME,
            self.CLEAN_AGENT_STREET,
            self.CLEAN_AGENT_ZIP,
            self.CLEAN_OFFICE_STREET,
            self.CLEAN_OFFICE_CITY,
            self.CLEAN_OFFICE_ZIP,
        ]

    @property
    def name_clean(self):
        return [
            self.CLEAN_NAME,
            self.CLEAN_MANAGER_MEMBER_NAME,
            self.CLEAN_AGENT_NAME,
        ]

    @property
    def address_clean(self):
        return {
            "street": [
                self.CLEAN_MANAGER_MEMBER_STREET,
                self.CLEAN_AGENT_STREET,
                self.CLEAN_OFFICE_STREET,
            ],
            "zip": [
                self.CLEAN_MANAGER_MEMBER_ZIP,
                self.CLEAN_AGENT_ZIP,
                self.CLEAN_OFFICE_ZIP,
            ]
        }

    @property
    def clean_address_map(self):
        return [
            {
                "full_address": self.CLEAN_MANAGER_MEMBER_ADDRESS,
                "address_cols": [
                    self.CLEAN_MANAGER_MEMBER_STREET,
                    self.CLEAN_MANAGER_MEMBER_CITY,
                    self.CLEAN_MANAGER_MEMBER_ZIP,
                ]
            },
            {
                "full_address": self.CLEAN_AGENT_ADDRESS,
                "address_cols": [
                    self.CLEAN_AGENT_STREET,
                    self.CLEAN_AGENT_ZIP,
                ]
            },
            {
                "full_address": self.CLEAN_OFFICE_ADDRESS,
                "address_cols": [
                    self.CLEAN_OFFICE_STREET,
                    self.CLEAN_OFFICE_CITY,
                    self.CLEAN_OFFICE_ZIP,
                ]
            },
        ]

    @property
    def out(self):
        return [
            self.FILE_NUMBER,
            self.STATUS,
            self.RAW_NAME,
            self.RAW_MANAGER_MEMBER_NAME,
            self.RAW_MANAGER_MEMBER_STREET,
            self.RAW_MANAGER_MEMBER_CITY,
            self.RAW_MANAGER_MEMBER_ZIP,
            self.RAW_MANAGER_MEMBER_ADDRESS,
            self.RAW_AGENT_NAME,
            self.RAW_AGENT_STREET,
            self.RAW_AGENT_ZIP,
            self.RAW_AGENT_ADDRESS,
            self.RAW_OFFICE_STREET,
            self.RAW_OFFICE_CITY,
            self.RAW_OFFICE_ZIP,
            self.RAW_OFFICE_ADDRESS,
            self.CLEAN_NAME,
            self.CLEAN_MANAGER_MEMBER_NAME,
            self.CLEAN_MANAGER_MEMBER_STREET,
            self.CLEAN_MANAGER_MEMBER_CITY,
            self.CLEAN_MANAGER_MEMBER_ZIP,
            self.CLEAN_MANAGER_MEMBER_ADDRESS,
            self.CLEAN_AGENT_NAME,
            self.CLEAN_AGENT_STREET,
            self.CLEAN_AGENT_ZIP,
            self.CLEAN_AGENT_ADDRESS,
            self.CLEAN_OFFICE_STREET,
            self.CLEAN_OFFICE_CITY,
            self.CLEAN_OFFICE_ZIP,
            self.CLEAN_OFFICE_ADDRESS,
        ]

    @property
    def unvalidated_col_objs(self):
        return [
            {
                self.RAW_MANAGER_MEMBER_STREET: "raw_street",
                self.RAW_MANAGER_MEMBER_CITY: "raw_city",
                self.RAW_MANAGER_MEMBER_ZIP: "raw_zip",
                self.RAW_MANAGER_MEMBER_ADDRESS: "raw_address",
                self.CLEAN_MANAGER_MEMBER_STREET: "clean_street",
                self.CLEAN_MANAGER_MEMBER_CITY: "clean_city",
                self.CLEAN_MANAGER_MEMBER_ZIP: "clean_zip",
                self.CLEAN_MANAGER_MEMBER_ADDRESS: "clean_address",
            },
            {
                self.RAW_AGENT_STREET: "raw_street",
                self.RAW_AGENT_ZIP: "raw_zip",
                self.RAW_AGENT_ADDRESS: "raw_address",
                self.CLEAN_AGENT_STREET: "clean_street",
                self.CLEAN_AGENT_ZIP: "clean_zip",
                self.CLEAN_AGENT_ADDRESS: "clean_address",
            },
            {
                self.RAW_OFFICE_STREET: "raw_street",
                self.RAW_OFFICE_CITY: "raw_city",
                self.RAW_OFFICE_ZIP: "raw_zip",
                self.RAW_OFFICE_ADDRESS: "raw_address",
                self.CLEAN_OFFICE_STREET: "clean_street",
                self.CLEAN_OFFICE_CITY: "clean_city",
                self.CLEAN_OFFICE_ZIP: "clean_zip",
                self.CLEAN_OFFICE_ADDRESS: "clean_address",
            }
        ]


class ColumnTaxpayerRecords(ColumnManagerBase):

    RAW_NAME_ADDRESS: str = "raw_name_address"  # todo: add documentation: this is the unique value to merge with the final property dataset
    RAW_NAME: str = "raw_name"
    RAW_ADDRESS: str = "raw_address"
    RAW_STREET: str = "raw_street"
    RAW_CITY: str = "raw_city"
    RAW_STATE: str = "raw_state"
    RAW_ZIP: str = "raw_zip"
    CLEAN_NAME_ADDRESS: str = "clean_name_address"
    CLEAN_NAME: str = "clean_name"
    CLEAN_ADDRESS: str = "clean_address"
    CLEAN_STREET: str = "clean_street"
    CLEAN_CITY: str = "clean_city"
    CLEAN_STATE: str = "clean_state"
    CLEAN_ZIP: str = "clean_zip"
    CORE_NAME: str = "core_name"
    IS_RENTAL: str = "is_rental"
    IS_BANK: str = "is_bank"
    IS_PERSON: str = "is_person"
    IS_COMMON_NAME: str = "is_common_name"
    IS_ORG: str = "is_org"
    IS_LLC: str = "is_llc"

    # merged address data
    # todo: determine whether it's necessary to include all these? or only include ones that are absolutely necessary for whatever is being done
    FORMATTED_ADDRESS: str = "formatted_address"

    @property
    def out(self):
        return [
            self.RAW_NAME,
            self.RAW_STREET,
            self.RAW_CITY,
            self.RAW_STATE,
            self.RAW_ZIP,
            self.RAW_ADDRESS,
            self.RAW_NAME_ADDRESS,
            self.CLEAN_NAME,
            self.CLEAN_STREET,
            self.CLEAN_CITY,
            self.CLEAN_STATE,
            self.CLEAN_ZIP,
            self.CLEAN_ADDRESS,
            self.CLEAN_NAME_ADDRESS,
        ]

    @property
    def unvalidated_addr_cols(self):
        return [
            self.RAW_STREET,
            self.RAW_CITY,
            self.RAW_STATE,
            self.RAW_ZIP,
            self.RAW_ADDRESS,
            self.CLEAN_STREET,
            self.CLEAN_CITY,
            self.CLEAN_STATE,
            self.CLEAN_ZIP,
            self.CLEAN_ADDRESS,
        ]


class ColumnProperties(ColumnManagerBase):

    PIN: str = "pin"
    RAW_NAME_ADDRESS: str = "raw_name_address"  # todo: add documentation: unique identifier to associate properties to taxpayer records should be RAW name+address concatenation
    CLEAN_NAME_ADDRESS: str = "clean_name_address"
    CLASS_CODE: str = "class_code"
    NUM_UNITS: str = "num_units"  # todo: add documentation: optional - ideal but will work without it

    @property
    def out(self):
        return [
            self.PIN,
            self.RAW_NAME_ADDRESS,
            self.CLEAN_NAME_ADDRESS,
            self.CLASS_CODE,
            self.NUM_UNITS,
        ]


class ColumnUnvalidatedAddrs(ColumnManagerBase):

    RAW_STREET: str = "raw_street"
    RAW_CITY: str = "raw_city"
    RAW_STATE: str = "raw_state"
    RAW_ZIP: str = "raw_zip"
    RAW_ADDRESS: str = "raw_address"
    CLEAN_STREET: str = "raw_street"
    CLEAN_CITY: str = "raw_city"
    CLEAN_STATE: str = "raw_state"
    CLEAN_ZIP: str = "raw_zip"
    CLEAN_ADDRESS: str = "raw_address"


class ColumnValidatedAddrs(ColumnManagerBase):

    RAW_ADDRESS: str = "raw_address"
    CLEAN_ADDRESS: str = "clean_address"
    NUMBER: str = "number"
    PREDIRECTIONAL: str = "predirectional"
    PREFIX: str = "prefix"
    STREET: str = "street"
    SUFFIX: str = "suffix"
    POSTDIRECTIONAL: str = "postdirectional"
    SECONDARYUNIT: str = "secondaryunit"
    SECONDARYNUMBER: str = "secondarynumber"
    CITY: str = "city"
    COUNTY: str = "county"
    STATE: str = "state"
    ZIP: str = "zip"
    COUNTRY: str = "country"
    LNG: str = "lng"
    LAT: str = "lat"
    ACCURACY: str = "accuracy"
    FORMATTED_ADDRESS: str = "formatted_address"


class ColumnAddressAnalysis(ColumnManagerBase):

    ADDRESS: str = "ADDRESS"
    COUNT: str = "COUNT"
    NAME: str = "NAME"
    URL: str = "URL"
    NOTES: str = "NOTES"
    IS_LANDLORD_ORG: str = "IS_LANDLORD_ORG"
    IS_GOVT_AGENCY: str = "IS_GOVT_AGENCY"
    IS_LAWFIRM: str = "IS_LAWFIRM"
    IS_MISSING_SUITE: str = "IS_MISSING_SUITE"
    IS_FINANCIAL_SERVICES: str = "IS_FINANCIAL_SERVICES"
    IS_ASSOC_BUS: str = "IS_ASSOC_BUS"
    FIX_ADDRESS: str = "FIX_ADDRESS"
    IS_FIXED: str = "IS_FIXED"
    IS_VIRTUAL_OFFICE_AGENT: str = "IS_VIRTUAL_OFFICE_AGENT"
    IS_NONPROFIT: str = "IS_NONPROFIT"
    IS_IGNORE_MISC: str = "IS_IGNORE_MISC"
    YELP_URL: str = "YELP_URL"
    GOOGLE_URL: str = "GOOGLE_URL"
