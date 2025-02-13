import abc
from enum import StrEnum

import pandera as pa
from pandas import CategoricalDtype


# test

class OPNDFModel(pa.DataFrameModel):
    pass


class OPNDFStrEnum(StrEnum):

    @classmethod
    def to_categorical_dtype(cls) -> CategoricalDtype:
        return CategoricalDtype(categories=[e.value for e in cls], ordered=False)
