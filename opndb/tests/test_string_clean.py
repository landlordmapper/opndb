import pandas as pd

from opndb.services.string_clean import CleanStringBase, CleanStringName


def test_string_cleaners():

    llc = "BLOOMINGTON  MN   55439"

    llc = CleanStringBase.make_upper(llc)
    llc = CleanStringBase.remove_symbols_punctuation(llc)
    llc = CleanStringBase.trim_whitespace(llc)
    llc = CleanStringBase.remove_extra_spaces(llc)
    llc = CleanStringBase.fix_llcs(llc)
    llc = CleanStringBase.deduplicate(llc)
    llc = CleanStringBase.take_first(llc)
    llc = CleanStringBase.combine_numbers(llc)
    llc = CleanStringName.switch_the(llc)

    assert llc == "NEWPORT PROPERTIES LLC"


def test_is_zip_irregular():
    def is_zip_irregular(zip_code: str | float) -> bool:
        if pd.isnull(zip_code):
            return True
        # zip code is as expected - either 5 or 9 numbers
        if len(zip_code) == 5 or len(zip_code) == 9:
            return False
        # zip code include final 4 digits separated by dash
        if "-" in zip_code:
            zip_split = zip_code.split("-")
            if len(zip_split[0]) == 5 and len(zip_split[1]) == 4:
                return False
            else:
                return True
        return True

    zip_code = "FRANCE"
    is_irregular = is_zip_irregular(zip_code)
    assert is_irregular == True