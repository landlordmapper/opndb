import abc
from enum import StrEnum
from typing import Any

import pandas as pd
import pandera as pa
from pandas import CategoricalDtype


class OPNDFModel(pa.DataFrameModel):
    pass
    # class Config:
    #     # Allow dataframes with additional columns
    #     strict = False
    #     # Use pandas' dtype_backend
    #     dtype_backend = "pandas"
    #     # Enable coercion for all child models
    #     coerce = True
    #
    #     @classmethod
    #     def coerce_dtype(cls, data: Any, dtype: Any) -> Any:
    #         if dtype == pd.Int64Dtype:
    #             # Handle conversion to Pandas nullable integer type
    #             if isinstance(data, pd.Series):
    #                 # First convert to numeric with coerce to handle strings
    #                 numeric_data = pd.to_numeric(data, errors='coerce')
    #                 # Then convert to Int64
    #                 return numeric_data.astype('Int64')
    #         # Fall back to default behavior for other types
    #         return pa.DataFrameModel.Config.coerce_dtype(data, dtype)


class OPNDFStrEnum(StrEnum):

    @classmethod
    def to_categorical_dtype(cls) -> CategoricalDtype:
        return CategoricalDtype(categories=[e.value for e in cls], ordered=False)
