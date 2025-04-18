import abc
from enum import StrEnum
from typing import Any

import pandas as pd
import pandera as pa
from pandas import CategoricalDtype


class OPNDFModel(pa.DataFrameModel):

    @classmethod
    def boolean_fields(cls, recursive: bool = False) -> list[str]:
        """
        Returns list of strings representing boolean column names for the dataset associated with the pandera model
        class
        """
        bool_fields = []
        if recursive:
            # Walk MRO (method resolution order) to include parent classes
            for base in cls.__mro__:
                if not hasattr(base, "__annotations__"):
                    continue
                for field_name, annotation in base.__annotations__.items():
                    if annotation is bool or annotation == "bool":
                        bool_fields.append(field_name)
        else:
            for field_name, annotation in cls.__annotations__.items():
                if annotation is bool or annotation == "bool":
                    bool_fields.append(field_name)
        return list(set(bool_fields))  # remove duplicates in case of overrides


class OPNDFStrEnum(StrEnum):

    @classmethod
    def to_categorical_dtype(cls) -> CategoricalDtype:
        return CategoricalDtype(categories=[e.value for e in cls], ordered=False)
