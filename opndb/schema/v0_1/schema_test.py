import pandera as pa


# --------------------
# ----RAW DATASETS----
# --------------------
class PropsTaxpayers(pa.DataFrameModel):
    """
    Raw dataset containing both property and taxpayer record data. The opndb workflow will split up this dataset into
    separate datasets: one for taxpayer records, the other for properties.
    """
    pin: str = pa.Field(
        nullable=False,
        unique=True,
        required=True,
        title="PIN",
        description="Unique tax identifier for the property",
    )
    tax_name: str = pa.Field(
        required=True,
        nullable=False,
        title="Taxpayer Name",
        description="Taxpayer name indicated for the property",
    )
    tax_street: str = pa.Field(
        required=True,
        nullable=False,
        title="Taxpayer Street",
        description="Street address for the taxpayer of the property, including street number, street name, secondary/unit number and all prefixes and suffixes.",
    )
    tax_city: str = pa.Field(
        required=True,
        nullable=False,
        title="Taxpayer City",
        description="City associated with property taxpayer's street address."
    )
    tax_state: str = pa.Field(
        required=True,
        nullable=False,
        title="Taxpayer State",
        description="State associated with property taxpayer's street address."
    )
    tax_zip: str = pa.Field(
        required=True,
        nullable=False,
        title="Taxpayer Zip",
        description="Zip code associated with property taxpayer's street address."
    )
    # tax_address: str = pa.Field(required=False)
    class_code: str = pa.Field(
        nullable=False,
        required=True,
        title="Class Code",
        description="Municipal code indicating land use for the property, required for subsetting rental properties."
    )
    num_units: int = pa.Field(
        nullable=True,
        required=False,
        title="Number of Units",
        description="Number of rental apartment units in the property."
    )

class Corps(pa.DataFrameModel):
    name: str = pa.Field()
    file_number: str = pa.Field()
    date_incorporated: str = pa.Field()
    date_dissolved: str = pa.Field()
    status: str = pa.Field()
    president_name: str = pa.Field()
    president_address: str = pa.Field()
    president_street: str = pa.Field()
    president_city: str = pa.Field()
    president_state: str = pa.Field()
    president_zip: str = pa.Field()
    secretary_name: str = pa.Field()
    secretary_address: str = pa.Field()
    secretary_street: str = pa.Field()
    secretary_city: str = pa.Field()
    secretary_state: str = pa.Field()
    secretary_zip: str = pa.Field()

class LLCs(pa.DataFrameModel):
    name: str = pa.Field()
    file_number: str = pa.Field()
    date_incorporated: str = pa.Field()
    date_dissolved: str = pa.Field()
    status: str = pa.Field()
    manager_name: str = pa.Field()
    manager_address: str = pa.Field()
    manager_street: str = pa.Field()
    manager_city: str = pa.Field()
    manager_state: str = pa.Field()
    manager_zip: str = pa.Field()
    agent_name: str = pa.Field()
    agent_address: str = pa.Field()
    agent_street: str = pa.Field()
    agent_city: str = pa.Field()
    agent_state: str = pa.Field()
    agent_zip: str = pa.Field()
    office_address: str = pa.Field()
    office_street: str = pa.Field()
    office_city: str = pa.Field()
    office_state: str = pa.Field()
    office_zip: str = pa.Field()

class ClassCodes(pa.DataFrameModel):
    code: str = pa.Field()
    category: str = pa.Field()
    description: str = pa.Field()
    is_rental: bool = pa.Field()

# -------------------------------
# ----DATASETS FOR PROCESSING----
# -------------------------------
class TaxpayerRecords(pa.DataFrameModel):
    raw_name: str = pa.Field(
        nullable=False,
        required=True,
        title="Raw Taxpayer Name",
        description="Taxpayer name EXACTLY how it appears in the raw data.",
    )
    raw_street: str = pa.Field(
        nullable=False,
        required=True,
        title="Raw Taxpayer Street",
        description="Taxpayer street address EXACTLY how it appears in the raw data."
    )
    raw_city: str = pa.Field(
        nullable=False,
        required=True,
        title="Raw Taxpayer City",
        description="Taxpayer city EXACTLY how it appears in the raw data."
    )
    raw_state: str = pa.Field(
        nullable=False,
        required=True,
        title="Raw Taxpayer State",
        description="Taxpayer state EXACTLY how it appears in the raw data."
    )
    raw_zip: str = pa.Field(
        nullable=False,
        required=True,
        title="Raw Taxpayer Zip",
        description="Taxpayer zip code EXACTLY how it appears in the raw data."
    )
    raw_address: str = pa.Field(
        nullable=False,
        required=True,
        title="Raw Taxpayer Address",
        description="Concatenation of raw taxpayer address components."
    )
    raw_name_address: str = pa.Field(
        nullable=False,
        required=True,
        title="Raw Taxpayer Name+Address",
        description="Concatenation of raw taxpayer name and full address, to be used for identifying unique taxpayer records."
    )
    clean_name: str = pa.Field(
        nullable=False,
        required=True,
        title="Clean Taxpayer Name",
        description="Taxpayer name AFTER running it through the string cleaners."
    )
    clean_street: str = pa.Field(
        nullable=False,
        required=True,
        title="Clean Taxpayer Street",
        description="Taxpayer street address AFTER running it through the string cleaners."
    )
    clean_city: str = pa.Field(
        nullable=False,
        required=True,
        title="Clean Taxpayer City",
        description="Taxpayer city AFTER running it through the string cleaners."
    )
    clean_state: str = pa.Field(
        nullable=False,
        required=True,
        title="Clean Taxpayer State",
        description="Taxpayer state AFTER running it through the string cleaners."
    )
    clean_zip: str = pa.Field(
        nullable=False,
        required=True,
        title="Clean Taxpayer Zip",
        description="Taxpayer zip code AFTER running it through the string cleaners."
    )
    clean_address: str = pa.Field(
        nullable=False,
        required=True,
        title="Clean Taxpayer Address",
        description="Concatenation of clean taxpayer address components."
    )

class TaxpayerRecordsMerged(TaxpayerRecords):
    """
    Outputted taxpayer record dataset resulting from the AddressMerge workflow
    """
    clean_address_v: str = pa.Field(
        nullable=True,
        required=True,
        title="Validated Taxpayer Address",
        description="Validated clean taxpayer mailing address."
    )

class TaxpayersFixed(TaxpayerRecordsMerged):
    is_common_name: bool = pa.Field(
        nullable=False,
        required=True,
        title="Is Common Name?",
        description="Boolean column indicating whether or not the taxpayer name is identified as a 'common name' (ex: John Smith, Juan Garcia, etc.)."
    )
    is_landlord_org: bool = pa.Field(
        nullable=False,
        required=True,
        title="Is Landlord Org?",
        description="Boolean column indicating whether or not the validated taxpayer address is associated with a 'landlord organization' (property management company, wealth management company, realtor, etc.)."
    )

class TaxpayersSubsetted(TaxpayersFixed):
    is_rental: bool = pa.Field(
        nullable=False,
        required=True,
        title="Is Rental?",
        description="Boolean column indicating whether or not the taxpayer record is associated with a rental property."
    )

class TaxpayersPrepped(TaxpayersSubsetted):
    core_name: str = pa.Field()
    is_trust: bool = pa.Field()
    is_person: bool = pa.Field()
    is_org: bool = pa.Field()
    is_llc: bool = pa.Field()
    entity_clean_name: str = pa.Field()
    entity_core_name: str = pa.Field()
    merge_address_1: str = pa.Field()
    merge_address_2: str = pa.Field()
    merge_address_3: str = pa.Field()
    is_clean_match: bool = pa.Field()
    is_core_match: bool = pa.Field()
    is_string_match: bool = pa.Field()

class TaxpayersStringMatched(TaxpayersPrepped):
    match_address: str = pa.Field()
    clean_name_address: str = pa.Field()
    core_name_address: str = pa.Field()
    include_address: bool = pa.Field()
    string_matched_name_1: str = pa.Field()
    string_matched_name_2: str = pa.Field()
    string_matched_name_3: str = pa.Field()

class TaxpayersNetworked(TaxpayersStringMatched):
    network_1: str = pa.Field()
    network_1_short: str = pa.Field()
    network_2: str = pa.Field()
    network_2_short: str = pa.Field()
    network_3: str = pa.Field()
    network_3_short: str = pa.Field()
    network_4: str = pa.Field()
    network_4_short: str = pa.Field()
    network_5: str = pa.Field()
    network_5_short: str = pa.Field()
    network_6: str = pa.Field()
    network_6_short: str = pa.Field()

class UnvalidatedAddrs(pa.DataFrameModel):
    raw_street: str = pa.Field()
    raw_city: str = pa.Field()
    raw_state: str = pa.Field()
    raw_zip: str = pa.Field()
    raw_address: str = pa.Field()
    clean_street: str = pa.Field()
    clean_city: str = pa.Field()
    clean_state: str = pa.Field()
    clean_zip: str = pa.Field()
    clean_address: str = pa.Field()
    addr_type: str = pa.Field()
    origin: str = pa.Field()
    active: bool = pa.Field(
        title="Active?",
        description="Boolean column representing whether or not the corporate or LLC record from which the address originates is active or inactive."
    )

class UnvalidatedAddrsClean(UnvalidatedAddrs):
    is_pobox: bool = pa.Field()

class GcdValidated(pa.DataFrameModel):
    pass

class GcdUnvalidated(pa.DataFrameModel):
    pass


# --------------------------
# ----OUTPUTTED DATASETS----
# --------------------------
# these will be the final outputted datasets to be stored in the s3 bucket and available to be pulled down and used
