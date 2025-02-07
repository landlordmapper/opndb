import pandas as pd
import pandera as pa
from opndb.validator.df_model import OPNDFModel
from pandera.typing import Index, DataFrame, Series


class Building(OPNDFModel):
    pin: Series[str] = pa.Field(description="Property PIN / unique tax ID")
    year_built: Series[int] = pa.Field(lt=3000, default=None, nullable=True, coerce=True, description="Year building was built")

df = pd.DataFrame({
    "pin": ["2001abc", "1232002", "qqqq3"],
    "year_built": [1996, "1997", "2024"],
})


print(Building.validate(df))

bad_df = pd.DataFrame({
    "pin": ["2001abc", "1232002", "qqqq3"],
    "year_built": ["3001", "1997", "2024"],
})

try:
    print(Building.validate(bad_df))
except pa.errors.SchemaError as e:
    print(f"bad dataframe failed as expected: {e}")
    
