import pandas as pd

from opndb.constants.base import DIRECTIONS, STREET_SUFFIXES
from opndb.services.string_clean import CleanStringBase, CleanStringName, CleanStringAddress


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


def test_convert_nsew():

    def convert_nsew(text: str) -> str:
        """
            'NORTH MAIN STREET' -> 'N MAIN STREET'
            '123 SOUTH WEST AVE' -> '123 S WEST AVE'
            'EAST 42ND STREET' -> 'E 42ND STREET'
            'NORTH WEST PLAZA' -> 'N WEST PLAZA'
            "SOUTH WEST AVE" > “S WEST AVE”
            "NORTH EAST ST" > “N EAST ST”
            "NORTH WEST BUILDING" > “NORTH WEST BUILDING”
        """
        try:
            words = text.split()
            i = 0
            while i < len(words):
                if words[i] in DIRECTIONS:
                    if i + 1 < len(words) and words[i + 1] in DIRECTIONS:
                        if i + 2 < len(words) and words[i + 2] in STREET_SUFFIXES.values():
                            words[i] = DIRECTIONS[words[i]]  # Convert only the first direction
                            i += 1  # Skip modifying the second direction
                        else:
                            i += 1  # Skip conversion if no street suffix follows
                    else:
                        words[i] = DIRECTIONS[words[i]]
                i += 1

            return " ".join(words)
        except:
            return text

    text = "NORTH MAIN STREET"

    text_cleaned = convert_nsew(text)

    assert text_cleaned == "N MAIN STREET"