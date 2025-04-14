import pandera as pa

from opndb.validator.df_model import OPNDFModel


class PropsTaxpayers(OPNDFModel):
    """
    Raw dataset containing both property and taxpayer record data. The opndb workflow will split up this dataset into
    separate datasets: one for taxpayer records, the other for properties.
    """
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
    # tax_address: str = pa.Field()
    class_code: str = pa.Field(
        nullable=False,
        title="Class Code",
        description="Municipal code indicating land use for the property, required for subsetting rental properties."
    )
    num_units: int = pa.Field(
        nullable=True,
        title="Number of Units",
        description="Number of rental apartment units in the property."
    )

class Corps(OPNDFModel):
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
    date_incorporated: str = pa.Field(
        nullable=True,
        unique=False,
        title="Date Incorporated",
        description="Date Incorporated",
    )
    date_dissolved: str = pa.Field(
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
    president_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="President Name",
        description="President name",
    )
    president_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="President Address",
        description="President address",
    )
    president_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="President Street",
        description="President street",
    )
    president_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="President City",
        description="President city",
    )
    president_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="President State",
        description="President state",
    )
    president_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="President Zip Code",
        description="President zip code",
    )
    secretary_name: str = pa.Field(
        nullable=False,
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
    secretary_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Street",
        description="Secretary street",
    )
    secretary_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary City",
        description="Secretary city",
    )
    secretary_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary State",
        description="Secretary state",
    )
    secretary_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="Secretary Zip Code",
        description="Secretary zip code",
    )

class LLCs(OPNDFModel):
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
    date_incorporated: str = pa.Field(
        nullable=True,
        unique=False,
        title="Date Incorporated",
        description="Date Incorporated",
    )
    date_dissolved: str = pa.Field(
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
    manager_name: str = pa.Field(
        nullable=False,
        unique=False,
        title="Manager/Member Name",
        description="Manager/member name",
    )
    manager_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Address",
        description="Manager/member address",
    )
    manager_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member Street",
        description="Manager/member street",
    )
    manager_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member City",
        description="Manager/member city",
    )
    manager_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Manager/Member State",
        description="Manager/member state",
    )
    manager_zip: str = pa.Field(
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
    agent_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Address",
        description="Agent address",
    )
    agent_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Street",
        description="Agent street",
    )
    agent_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Agent City",
        description="Agent city",
    )
    agent_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Agent State",
        description="Agent state",
    )
    agent_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="Agent Zip Code",
        description="Agent zip code",
    )
    office_address: str = pa.Field(
        nullable=True,
        unique=False,
        title="Office Address",
        description="Office address",
    )
    office_street: str = pa.Field(
        nullable=True,
        unique=False,
        title="Office Street",
        description="Office street",
    )
    office_city: str = pa.Field(
        nullable=True,
        unique=False,
        title="Office City",
        description="Office city",
    )
    office_state: str = pa.Field(
        nullable=True,
        unique=False,
        title="Office State",
        description="Office state",
    )
    office_zip: str = pa.Field(
        nullable=True,
        unique=False,
        title="Office Zip Code",
        description="Office zip code",
    )

class ClassCodes(OPNDFModel):
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