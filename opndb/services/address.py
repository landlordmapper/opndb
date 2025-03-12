import re
from concurrent.futures import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
import time
import traceback

import pandas as pd
import requests as req

from opndb.constants.base import GEOCODIO_URL, PO_BOXES_DEPT, PO_BOXES_REMOVE, PO_BOXES
from opndb.services.config import ConfigManager
from opndb.services.string_clean import CleanStringBase as clean_base
from opndb.types.base import (
    CleanAddress,
    GeocodioResult,
    GeocodioResponse,
    GeocodioResultProcessed,
    GeocodioResultFlat, GeocodioResultFinal, GeocodioReturnObject, WorkflowConfigs
)
from opndb.services.dataframe.base import (
    DataFrameOpsBase as ops_df,
    DataFrameBaseCleaners as clean_df
)
from opndb.utils import (UtilsBase as utils, PathGenerators as utils_path)


class AddressBase:

    # todo: get this out of here - all file processing should be handled in the workflow

    def __init__(self):
        self.geocodio_api_key = ""

    @classmethod
    def get_unique_addresses(cls, df: pd.DataFrame, addr_cols: CleanAddress):
        """Returns all unique addresses found in the dataframe."""
        pass

    @classmethod
    def format_pobox(cls):
        """Runs PO box formatter on raw address and adds to DATA_ROOT/processed/validated_addrs.csv"""
        pass

    @classmethod
    def save_unvalidated_addrs_initial(cls, dfs: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Saves initial unvalidated address file to DATA_ROOT/processed/unvalidated_addrs. Run only once, after the raw
        data has been cleaned & validated but BEFORE address validation.
        """
        pass

    @classmethod
    def save_validated_addrs_initial(cls, df: pd.DataFrame) -> str:
        return ops_df.save_df(df, cls.validated_addrs_path)

    @classmethod
    def call_geocodio(cls, api_key: str, address_search_string: str) -> list[GeocodioResult] | None:
        """Executes Geocodio API call."""
        if api_key == "":
            raise Exception("No API key detected.")
        url: str = GEOCODIO_URL + api_key + "&q=" + address_search_string
        res: req.Response = req.get(url)
        if res.status_code == 200:
            data: GeocodioResponse = res.json()
            return data["results"]
        else:
            return None

    @classmethod
    def save_geocodio_partial(cls, results: list[dict], configs: WorkflowConfigs):
        timestamp = utils.get_timestamp()
        df_partial = pd.DataFrame(results)
        ops_df.save_df(df_partial, utils_path.geocodio_partial(configs, f"gcd_partial_({timestamp}).csv"))

    @classmethod
    def check_matching_addrs(cls, df: pd.DataFrame) -> bool:
        """
        Check whether the formatted addresses in the GeocodioResult objects are identical. Returns 'True' if they are,
        'False' if they aren't.
        """
        formatted_addrs = set(df["formatted_address"].tolist())  # todo: add 'formatted_address' column to dataframe
        return len(formatted_addrs) == 1

    @classmethod
    def flatten_geocodio_result(cls, result: GeocodioResult) -> GeocodioResultFlat:
        """
        Flattens geocodio result by moving lat, lng, accuracy and formatted_address into the highest level key/value
        pair nesting.
        """
        address_components = result.get("address_components", {})
        location = result.get("location", {})
        return {
            "number": address_components.get("number", ""),
            "predirectional": address_components.get("predirectional", ""),
            "prefix": address_components.get("prefix", ""),
            "street": address_components.get("street", ""),
            "suffix": address_components.get("suffix", ""),
            "postdirectional": address_components.get("postdirectional", ""),
            "secondaryunit": address_components.get("secondaryunit", ""),
            "secondarynumber": address_components.get("secondarynumber", ""),
            "city": address_components.get("city", ""),
            "county": address_components.get("county", ""),
            "state": address_components.get("state", ""),
            "zip": address_components.get("zip", ""),
            "country": address_components.get("country", ""),
            "lng": location.get("lng", ""),
            "lat": location.get("lat", ""),
            "accuracy": result.get("accuracy", ""),
            "formatted_address": result.get("formatted_address", "")
        }

    # @classmethod
    # def get_flatted_geocodio_result_from_df(cls, row: pd.Series) -> GeocodioResultFlat:
    #     return {
    #         "number": row["number"],
    #         "predirectional": row["predirectional"],
    #         "prefix": row["prefix"],
    #         "street": row["street"],
    #         "suffix": row["suffix"],
    #         "postdirectional": row["postdirectional"],
    #         "secondaryunit": row["secondaryunit"],
    #         "secondarynumber": row["secondarynumber"],
    #         "city": row["city"],
    #         "county": row["county"],
    #         "state": row["state"],
    #         "zip": row["zip"],
    #         "country": row["country"],
    #         "lng": row["lng"],
    #         "lat": row["lat"],
    #         "accuracy": row["accuracy"],
    #         "formatted_address": row["formatted_address"],
    #     }

    @classmethod
    def apply_filters(cls, clean_address: CleanAddress, df: pd.DataFrame) -> pd.DataFrame | None:

        """Filters geocodio results by street number and zip code. Additional filters to be added in the future."""

        # extract individual pieces of raw address
        addr_raw_split = clean_address["complete_addr"].split(",")

        if "street" in clean_address.keys():
            number_raw: str = clean_address["street"].split()[0]
        else:
            number_raw = addr_raw_split[0].split()[0]

        if "zip" in clean_address.keys():
            zip_raw: str = clean_address["zip"]
        else:
            zip_raw = addr_raw_split[-1]

        # FILTER 1: missing street numbers
        # filter out results with no street number - if none have street number, return object as is
        df_street_no: pd.DataFrame = df[df["number"] != ""]
        if df_street_no.empty:
            return None
        if len(df_street_no) == 1 or cls.check_matching_addrs(df_street_no):
            return df_street_no.iloc[[0]]

        # FILTER 2: multiple street numbers
        # print("filtering street number...")
        df_number = df[df["number"] == number_raw]
        if df_number.empty:
            return df
        if len(df_number) == 1 or cls.check_matching_addrs(df_number):
            return df_number.iloc[[0]]

        # FILTER 3: zip code
        # print("filtering zip code...")
        df_zip = df_number[df_number["zip"] == zip_raw]
        if df_zip.empty:
            return df_number
        if len(df_zip) == 1 or cls.check_matching_addrs(df_zip):
            return df_zip.iloc[[0]]
        else:
            return df_zip

        # FILTER 3: street name
        # print("filtering street name...")
        # streets = list(df_zip["street"].unique())
        # df_street = pd.DataFrame()
        # for street in streets:
        #     street_split = street.split()
        #     if all(word in addr_sc_split for word in street_split):
        #         df_street = df_zip[df_zip[cls.va.GCD_STREET] == street]
        #         break
        # if df_street.empty:
        #     unvalidated_rows.extend(df_zip.to_dict("records"))
        #     return
        # if len(df_street) == 1 or cls.check_matching_addrs(df_street):
        #     validated_rows.append(df_street.iloc[0].to_dict())
        #     return

        # FILTER 4: city name
        # print("filtering city name...")
        # city_sc = df_street.iloc[0][cls.va.TAXPAYER_CITY]
        # df_city = df_street[df_street[cls.va.TAXPAYER_CITY] == city_sc]
        # if df_city.empty:
        #     unvalidated_rows.extend(df_street.to_dict("records"))
        #     return
        # if len(df_city) == 1 or cls.check_matching_addrs(df_city):
        #     validated_rows.append(df_city.iloc[0].to_dict())
        #     return

        # FILTER 5: predirectional
        # print("filtering predirectional...")
        # predirs = df_city[cls.va.GCD_PREDIRECTIONAL].unique().tolist()
        # df_predir = df_city[df_city.apply(lambda row: cls.contains_predirectional(
        #     row[cls.va.TAXPAYER_ADDRESS],
        #     predirs
        # ), axis=1)]
        # if df_predir.empty:
        #     unvalidated_rows.extend(df_city.to_dict("records"))
        #     return
        # if len(df_predir) == 1 or cls.check_matching_addrs(df_predir):
        #     validated_rows.append(df_predir.iloc[0].to_dict())
        #     return

        # FILTER 6: street suffix
        # print("filtering street suffix...")
        # df_suffix = df_predir[df_predir[cls.va.GCD_SUFFIX].isin(addr_sc_split)]
        # if df_suffix.empty:
        #     unvalidated_rows.extend(df_predir.to_dict("records"))
        #     return
        # if len(df_suffix) == 1 or cls.check_matching_addrs(df_suffix):
        #     validated_rows.append(df_suffix.iloc[0].to_dict())
        #     return
        # unvalidated_rows.extend(df_predir.to_dict("records"))

    @classmethod
    def process_geocodio_results(
        cls,
        clean_address: CleanAddress,
        results: list[GeocodioResultFlat]
    ) -> GeocodioResultProcessed:

        """
        Processes results returned by call_geocodio(). Returns data object containing the raw address information, the
        original results object and a list of parsed results.
        """

        # instantiate GeocodioResultsProcessed object with empty results_parsed
        results_processed: GeocodioResultProcessed = {
            "clean_address": clean_address,
            "results": results,
            "results_parsed": None,
        }

        # if no results found, return object as is
        if len(results) == 0:
            return results_processed

        # convert results to df and perform basic cleaning
        df_results: pd.DataFrame = pd.DataFrame(results, dtype=str)
        df_results = df_results.fillna("")
        df_results = clean_df.make_upper(df_results)

        # run remaining results through filters
        df_filtered = cls.apply_filters(clean_address, df_results)

        if len(df_filtered) == 1:
            results_processed["results_parsed"] = [df_filtered.iloc[0].to_dict()]
        elif len(df_filtered) > 1:
            results_processed["results_parsed"] = df_filtered.to_dict("records")

        return results_processed

    @classmethod
    def run_geocodio(cls, api_key: str, df_addrs: pd.DataFrame, addr_col: str, configs: WorkflowConfigs, interval: int = 50) -> GeocodioReturnObject:
        """
        Executes Geocodio API calls for raw, unvalidated addresses.

        :param api_key: user's unique Geocodio API key
        :param df_addrs: Dataframe of rows containing raw address data, with one row per unique address.
        :param addr_col: Column name containing the addresses to be validated.
        :param interval: Optional interval setting to control how many Geocodio calls should trigger a partial save.
        :return:
        """
        return_obj: GeocodioReturnObject = {  # object to be used to create/update dataframes in workflow process
            "validated": [],
            "unvalidated": [],
            "failed": [],
        }
        try:
            gcd_results = []  # list of all results to store as partials
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=10) as executor:
                # store all unique addresses from dataframe into future object
                futures = {}
                for i, row in df_addrs.iterrows():
                    future = executor.submit(cls.call_geocodio, api_key, row[addr_col])
                    futures[future] = (i, row)
                # loop through futures object, executing geocodio calls for each one
                for future in as_completed(futures):  # todo: add progress bar visualization
                    try:
                        i, row = futures[future]
                        clean_address: CleanAddress = row.to_dict()  # create CleanAddress object from dataframe row
                        results: list[GeocodioResult] = future.result()  # fetch results returned by call_geocodio()
                        # flatten geocodio results to include lat, lng, accuracy and formatted address
                        flattened_results: list[GeocodioResultFlat] = [cls.flatten_geocodio_result(result) for result in results]
                        if results:  # API call succeeded, begin processing results
                            results_processed: GeocodioResultProcessed = cls.process_geocodio_results(
                                clean_address,
                                flattened_results
                            )
                            if len(results_processed["results_parsed"]) == 1:
                                new_validated = results_processed["results_parsed"][0]
                                new_validated["clean_address"] = clean_address["clean_address"]
                                return_obj["validated"].append(new_validated)
                            else:
                                for result in results_processed["results_parsed"]:
                                    new_unvalidated = result
                                    new_unvalidated["clean_address"] = clean_address["clean_address"]
                            # add all results and their associated raw address to the partial
                            for result in flattened_results:
                                gcd_results.append({**clean_address, **result})
                        else:
                            return_obj["failed"].append(clean_address)
                            gcd_results.append(clean_address)
                        # save geocodio partial and empty out gcd_results
                        if interval is not None and len(gcd_results) >= interval:
                            cls.save_geocodio_partial(gcd_results, configs)
                            gcd_results = []
                    except Exception as e:
                        print(f"Error: {e}")
                        print(traceback.format_exc())
                if interval is not None and gcd_results:
                    cls.save_geocodio_partial(gcd_results, configs)
            if interval is None:
                cls.save_geocodio_partial(gcd_results, configs)
            # log time
            end_time = time.time()
            print(f"Elapsed time: {round((end_time - start_time), 2)} minutes")
            return return_obj
        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())
            return return_obj

    @classmethod
    def add_to_validated_addrs(cls):
        """
        Saves validated addresses from 'DATA_ROOT/geocodio/gcd_validated.csv' to
        'DATA_ROOT/processed/validated_addrs.csv'.
        """
        pass

    @classmethod
    def removes_from_unvalidated_addrs(cls):
        """Removes validated addresses from 'DATA_ROOT/processed/unvalidated_addrs.csv'"""
        pass



    @classmethod
    def get_full_address(cls, row: pd.Series, address_cols: list[str]) -> str:
        address_parts = [
            str(row[col])
            for col in address_cols
            if pd.notna(row[col]) and str(row[col])
        ]
        return ", ".join(address_parts)

    @classmethod
    def fix_pobox(cls, address: str) -> str:
        """
        Detects variations of PO box addresses in raw taxpayer data and return standardized "PO BOX" format.
        """
        if not clean_base.get_is_pobox(address):
            return address

        raw_addr_split = address.split(",")

        if "#" in raw_addr_split[0]:
            raw_addr_no_spaces = raw_addr_split[0].replace(" ", "")
            match = re.search(r"([a-zA-Z])#", raw_addr_no_spaces)
            if match:
                raw_addr_cleaned = address.replace("#", "").strip().replace(" ", "")
            else:
                raw_addr_cleaned = address.replace("#", "DEPT ").strip().replace(" ", "")
        else:
            raw_addr_cleaned = address.replace(" ", "").strip()

        for po in PO_BOXES:

            if raw_addr_cleaned.startswith(po):

                raw_addr_stripped = raw_addr_cleaned[len(po):].strip()
                pobox_num = raw_addr_stripped.split(",")[0].strip()

                for remove in PO_BOXES_REMOVE:
                    if remove in pobox_num:
                        pobox_num = pobox_num.replace(remove, "")

                dep = ""
                dep_start = len(pobox_num)
                for dep_string in PO_BOXES_DEPT:
                    if dep_string in pobox_num:
                        dep_start = pobox_num.find(dep_string)
                        dep_stripped = pobox_num[dep_start + len(dep_string):].strip()
                        digits = "".join(filter(str.isdigit, dep_stripped))
                        dep = f"TAX DEPT {digits}"  # Insert space before department string
                        break

                # Add a space between the PO box number and department part
                raw_addr_split[0] = f"PO BOX {pobox_num[:dep_start].strip()} {dep}".strip()
                raw_addr_fixed = ",".join(raw_addr_split)
                return raw_addr_fixed


class AddressValidatorBase(AddressBase):

    """Handles operations that add to and remove from the master validated & unvalidated address files."""

    def __init__(self):
        super().__init__()
        self.df_validated: pd.DataFrame = ops_df.load_df(self.validated_addrs_path, str)
        self.df_unvalidated: pd.DataFrame = ops_df.load_df(self.unvalidated_addrs_path, str)

    def add_to_df_validated(self):
        # pass row of validated address data - pandera object?
        pass

    def remove_from_df_unvalidated(self):
        # pass index of row to be removed as argument
        pass


    def save(self) -> dict[str, str]:
        validated_path = ops_df.save_df(self.df_validated, self.validated_addrs_path)
        unvalidated_path = ops_df.save_df(self.df_unvalidated, self.unvalidated_addrs_path)
        return {
            "validated_path": validated_path,
            "unvalidated_path": unvalidated_path,
        }

