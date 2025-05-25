import pandera as pa
from opndb.validator.df_model import OPNDFModel


class UnvalidatedAddrs(OPNDFModel):
    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    raw_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Street",
        description="Raw street address (street number, street name and secondary address information)",
    )
    raw_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw City",
    )
    raw_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw State",
    )
    raw_zip_code: str = pa.Field(
        nullable=True,
        unique=False,
        title="Raw Zip Code",
    )
    raw_address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Address",
        description="Concatenated mailing address as appears in the raw data. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    clean_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Street",
        description="Clean street address (street number, street name and secondary address information)",
    )
    clean_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean City",
    )
    clean_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean State",
    )
    clean_zip_code: str = pa.Field(
        nullable=True,
        unique=False,
        title="Clean Zip Code",
    )
    clean_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Raw Address",
        description="Concatenated and cleaned full mailing address. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    addr_type: str = pa.Field(
        nullable=True,
        unique=False,
        title="Address Type",
    )
    origin: str = pa.Field(
        nullable=True,
        unique=False,
        title="Origin",
    )
    active: str = pa.Field(
        nullable=True,
        unique=False,
        title="Active?",
        description="Boolean column representing whether or not the corporate or LLC record from which the address originates is active or inactive."
    )


class UnvalidatedAddrsClean(UnvalidatedAddrs):
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _GEOCODIO_COLUMNS: list[str] = [
        "raw_address",
        "clean_address",
        "clean_street",
        "clean_city",
        "clean_state",
        "clean_zip",
        "is_pobox",
    ]

    @classmethod
    def geocodio_columns(cls) -> list[str]:
        return cls._GEOCODIO_COLUMNS

    is_pobox: bool | None = pa.Field(
        nullable=False,
        unique=False,
        title="Is PO Box?",
        description="Boolean indicating whether the street address has been identified as a PO Box."
    )


class Geocodio(OPNDFModel):
    # ---------------------------
    # ----COLUMN NAME OBJECTS----
    # ---------------------------
    _VALIDATED_COLUMNS: list[str] = [
        "raw_address",
        "clean_address",
        "number",
        "predirectional",
        "prefix",
        "street",
        "suffix",
        "postdirectional",
        "secondaryunit",
        "secondarynumber",
        "city",
        "county",
        "state",
        "zip",
        "country",
        "lng",
        "lat",
        "accuracy",
        "formatted_address",
    ]

    @classmethod
    def validated_columns(cls) -> list[str]:
        return cls._VALIDATED_COLUMNS

    # --------------------
    # ----MODEL FIELDS----
    # --------------------
    raw_address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Address",
        description="Concatenated mailing address as appears in the raw data. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    clean_address: str = pa.Field(
        nullable=False,
        unique=False,  # todo: this should be true
        title="Clean Address",
        description="Concatenated and cleaned full mailing address. Required regardless of whether the address is split into separate components (street, city, state, zip)."
    )
    number: str = pa.Field(
        nullable=False,
        unique=False,
        title="Street Number",
    )
    predirectional: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Predirectional",
        description="Predirectional value for street address (Ex: 'N' in '123 N OAK ST')"
    )
    prefix: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Prefix",
        description="Prefix for street address (Ex: 'AVE' in '4754 AVE H')"
    )
    street: str = pa.Field(
        nullable=False,
        unique=False,
        title="Street Name",
    )
    suffix: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Suffix",
        description="Suffix for street address (Ex: 'ST' in '123 OAK ST')"
    )
    postdirectional: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Postdirectional",
        description="Postdirectional value for street address (Ex: 'N' in '123 OAK ST N')"
    )
    secondary_unit: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Secondary Unit",
        description="Descriptor for secondary address (Ex: 'SUITE' in 'SUITE 101')"
    )
    secondary_number: str = pa.Field(
        nullable=True,
        unique=False,
        title="Street Secondary Number",
        description="Number component of the secondary address (Ex: '101' in 'SUITE 101')"
    )
    city: str = pa.Field(
        nullable=False,
        unique=False,
        title="City",
    )
    county: str = pa.Field(
        nullable=True,
        unique=False,
        title="County",
    )
    state: str = pa.Field(
        nullable=False,
        unique=False,
        title="State",
    )
    zip_code: str = pa.Field(
        nullable=False,
        unique=False,
        title="Zip Code",
    )
    country: str = pa.Field(
        nullable=False,
        unique=False,
        title="Country",
    )
    lng: str = pa.Field(
        nullable=True,
        unique=False,
        title="Longitude",
    )
    lat: str = pa.Field(
        nullable=True,
        unique=False,
        title="Latitude",
    )
    accuracy: str = pa.Field(
        nullable=True,
        unique=False,
        title="Accuracy",
        description="Accuracy score provided by Geocodio"
    )
    formatted_address: str = pa.Field(
        nullable=False,
        unique=False,
        title="Formatted Address",
        description="Concatenation of all address components into a single string."
    )
    is_pobox: bool | None = pa.Field(
        nullable=False,
        unique=False,
        title="Is PO Box?",
        description="Boolean indicating whether the street address has been identified as a PO Box."
    )


class GeocodioFormatted(Geocodio):
    formatted_address_v1: str = pa.Field()
    formatted_address_v2: str = pa.Field()
    formatted_address_v3: str = pa.Field()
    formatted_address_v4: str = pa.Field()


class FixingAddrs(Geocodio):
    check_sec_num: str = pa.Field(
        nullable=True,
        unique=False,
        title="Check Secondary Number",
        description="Contains secondary number value detected by regex run on street address for rows whose validated address does NOT contain a secondary number. Used to manually fix validated addresses that incorrectly excluded secondary address information."
    )


class FixingTaxNames(OPNDFModel):
    raw_value: str = pa.Field(
        nullable=False,
        unique=True,
        title="Raw Value",
        description="Value of taxpayer name to be changed"
    )
    standardized_value: str = pa.Field(
        nullable=False,
        unique=False,
        title="Standardized Value",
        description="New value of taxpayer name to be changed"
    )


class AddressAnalysis(OPNDFModel):
    address: str = pa.Field(
        nullable=False,
        unique=True,
        title="Address",
        description="Validated address to be analyzed"
    )
    frequency: str = pa.Field(
        nullable=False,
        unique=False,
        title="Frequency",
        description="Number of times the validated address appears in either the taxpayer, corporate or LLC datasets"
    )
    value: str | None = pa.Field(
        title="Value",
        description="Column used to match researched addresses for Chicago. Should NOT be present in future iterations"
    )
    name: str = pa.Field(
        nullable=True,
        unique=False,
        title="Name",
        description="Name of organization/entity associated with the address"
    )
    urls: str = pa.Field(
        nullable=True,
        unique=False,
        title="URLs",
        description="URLs associated with organization/entity associated with the address"
    )
    notes: str = pa.Field(
        nullable=True,
        unique=False,
        title="Notes",
        description="Any notes or additional considerations observed from researching address"
    )
    is_landlord_org: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Landlord Organization?",
        description="Boolean representing whether or not the entity/organization associated with the address is a property management company, real estate developer, real estate agency, investment or wealth management firm, or any other organization that can be held accountable for the conditions of a rental property. "
    )
    is_govt_agency: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Government Agency?",
        description="Boolean representing whether or not the address is associated with a government office"
    )
    is_lawfirm: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Lawfirm?",
        description="Boolean representing whether or not the entity/organization associated with the address is a lawfirm"
    )
    is_missing_suite: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Missing Suite?",
        description="Boolean representing whether or not the address is missing a suite number. Applies to addresses that point to office buildings, UPS stores, lock box services or other virtual mail services but that do NOT include a secondary address identifier. Could indicate that the address represents the owner of the commercial property itself, or a specific organization within the commercial property."
    )
    is_problematic_suite: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Problematic Suite?",
        description="Boolean representing whether or not the suite number in a researched address is either associated with too many different organizations to definitively establish a relationship, or is simply no information online could be found for that suite number."
    )
    is_realtor: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Realtor?",
        description="Boolean representing whether or not the address is associated with a realtor or a realty agency's office."
    )
    is_financial_services: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Financial Services?",
        description="Boolean representing whether or not the entity/organization associated with the address is a financial services company (tax services, mortgage services, etc.)"
    )
    is_assoc_bus: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Associated Business?",
        description="Boolean representing whether or not the entity/organization associated with the address is a business that is unrelated to property management. This could either be the business of the property's 'true owner', or the owner of the property containing the business."
    )
    fix_address: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Fix Address?",
        description="Indicates whether or not the address needs to be fixed"
    )
    is_virtual_office_agent: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Virtual Office / Agent?",
        description="Boolean representing whether or not the entity/organization associated with the address is a virtual office or registered agent service."
    )
    yelp_urls: str = pa.Field(
        nullable=True,
        unique=False,
        title="Yelp URLs",
        description="Yelp URLs associated with organization/entity associated with the address"
    )
    is_nonprofit: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Boolean representing whether or not the entity/organization associated with the address is a nonprofit organization",
        description=""
    )
    google_urls: str = pa.Field(
        nullable=True,
        unique=False,
        title="Google URLs",
        description="Google URLs associated with organization/entity associated with the address"
    )
    is_ignore_misc: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Ignore Misc?",
        description=""
    )
    google_place_id: str = pa.Field(
        nullable=True,
        unique=False,
        title="Google Place ID",
    )
    is_researched: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Is Researched?",
        description="Boolean representing whether or not the address has been manually researched"
    )


class FrequentTaxNames(OPNDFModel):
    value: str = pa.Field(
        nullable=False,
        unique=True,
        title="Value",
        description="Value of taxpayer name to be analyzed"
    )
    frequency: str = pa.Field(
        nullable=False,
        unique=False,
        title="Frequency",
        description="Number of times the name appears in the taxpayer dataset"
    )
    exclude_name: bool = pa.Field(
        nullable=True,
        unique=False,
        title="Exclude Name?",
        description="Boolean indicating whether or not the name is a common name and should therefore be excluded from the network graph generation."
    )
