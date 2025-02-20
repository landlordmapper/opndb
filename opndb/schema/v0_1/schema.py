import re
from enum import StrEnum
from typing import Final

import pandas as pd
import pandera as pa
from pandas import CategoricalDtype
from pandera import DateTime

from opndb.validator.df_model import OPNDFModel, OPNDFStrEnum
from pandera.typing import Series

VALID_ZIP_CODE_REGEX: Final[re] = r"^\d{5}(-\d{4})?$"


class CodeViolationStatus(OPNDFStrEnum):
    OPEN = "open"
    CLOSED = "closed"

class Building(OPNDFModel):
    """
    A standing structure that people live in
    """
    pin: Series[str] = pa.Field(description="Property PIN / unique tax ID")
    class_code: Series[str] = pa.Field(
        description="Property class code"
    )  # todo: need to know what format, is this cross-city, etc
    taxpayer_name: Series[str] = pa.Field(description="Name of the taxpayer")
    taxpayer_address: Series[str] = pa.Field(description="Address of the taxpayer")
    units_quantity: Series[int] = pa.Field(
        description="Number of units in the building"
    )
    year_built: Series[int] = pa.Field(
        lt=3000,
        default=None,
        nullable=True,
        coerce=True,
        description="Year building was built",
    )


class Address(OPNDFModel):
    street_number: Series[str] = pa.Field(description="Street number")
    street_name: Series[str] = pa.Field(description="Street name (excluding number)")
    street_suffix: Series[str] = pa.Field(
        description="Street suffix (e.g., St, Ave, Blvd)"
    )
    unit: Series[str] = pa.Field(
        nullable=True, description="Apartment, suite, or unit number"
    )
    city: Series[str] = pa.Field(description="City name")
    state: Series[str] = pa.Field(description="Two-letter USPS state abbreviation")
    zip_code: Series[str] = pa.Field(description="5-digit ZIP or ZIP+4")

    @pa.check("zip_code", regex=True, name="zip_code")
    def zip_code_regex(cls, a: Series[str]) -> Series[bool]:
        return a.str.match(VALID_ZIP_CODE_REGEX)


class Corporation(OPNDFModel):
    file_number: Series[str] = pa.Field(description="File number")  # todo: what is this
    creation_date: Series[DateTime] = pa.Field(
        description="Date of creation", nullable=True
    )
    president_name: Series[str] = pa.Field(
        description="Name of the corporation president", nullable=True
    )
    president_address: Series[str] = pa.Field(
        description="Address of the corporation president", nullable=True
    )
    secretary_name: Series[str] = pa.Field(
        description="Name of the corporation secretary", nullable=True
    )
    secretary_address: Series[str] = pa.Field(
        description="Address of the corporation secretary", nullable=True
    )


class LLC(OPNDFModel):
    file_number: Series[str] = pa.Field(description="File number")  # todo: what is this
    creation_Date: Series[DateTime] = pa.Field(
        description="Date of creation", nullable=True
    )
    manager_name: Series[str] = pa.Field(
        description="Name of the LLC manager", nullable=True
    )
    manager_address: Series[str] = pa.Field(
        description="Address of the LLC manager", nullable=True
    )
    agent_name: Series[str] = pa.Field(
        description="Name of the LLC agent", nullable=True
    )
    agent_address: Series[str] = pa.Field(
        description="Address of the LLC agent", nullable=True
    )
    office_address: Series[str] = pa.Field(
        description="Address of the office", nullable=True
    )


class PropertySale(OPNDFModel):
    property_pin: Series[str] = pa.Field(description="Property PIN / unique tax ID")
    sale_date: Series[DateTime] = pa.Field(description="Date of sale")
    sale_price: Series[int] = pa.Field(description="Sale price of the property")
    seller_name: Series[str] = pa.Field(description="Name of the seller")
    buyer_name: Series[str] = pa.Field(description="Name of the buyer")


class CodeViolation(OPNDFModel):
    date_opened: Series[DateTime] = pa.Field(
        description="Date the violation was opened"
    )
    date_updated: Series[DateTime] = pa.Field(
        description="Date the violation was updated", nullable=True
    )
    respondent_name: Series[str] = pa.Field(description="Name of respondant")
    fine_imposed: Series[int] = pa.Field(
        description="Fine imposed for the violation", nullable=True
    )
    liable: Series[bool] = pa.Field(
        description="Whether the respondent was found liable for the violation",
        nullable=True,
    )
    status: CategoricalDtype(categories=list(CodeViolationStatus)) = pa.Field(
        description="Status of the violation"
    )
