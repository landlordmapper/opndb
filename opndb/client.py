from typing import Final
import awswrangler as wr
from pandas import DataFrame

from opndb.validator.df_model import OPNDFModel

ROOT_S3_BUCKET_URI: Final[str] = 's3://opndb'

class _InternalOpnDBClient:
    """
    Internal class to handle S3 Operations after validation.

    Don't use it outside this module since we want to ALWAYS validate first.
    """

    root_s3_bucket_uri: str = ROOT_S3_BUCKET_URI
    staging_s3_uri: str = f'{root_s3_bucket_uri}/staging'
    testing_s3_uri: str = f'{root_s3_bucket_uri}/test'

    def store_dataframe(self, df: DataFrame, identifier: str):
        # Storing data on Data Lake
        df = DataFrame()
        wr.s3.to_parquet(
            df=df,
            path=self.testing_s3_uri + "/" + identifier,
            dataset=True,
        )



class OpnDBClient:

    def store_dataframe(self, df: DataFrame, model: OPNDFModel, identifier: str):
        """
        df: dataframe to store
        model: model to use to validate df and determine where it is stored
        identifier: unique identifier for the dataframe; todo this may be computed or something
        """
        # todo: validate df with model
        # todo: use model and identifier (or sth else) to determine where to store the dataframe
        pass

    def retrieve_dataframe(self, identifier: str):
        pass

if __name__ == "__main__":
    client = OpnDBClient()
    df = DataFrame([[1, 2], [3,4]])
    client.store_dataframe(df, 'test')