import re
import string

import numpy as np
import pandas as pd
import word2number as w2n

from opndb.constants.base import (
    DIRECTIONS,
    STREET_SUFFIXES,
    SECONDARY_KEYWORDS,
    CORE_NAME_KEYS,
    TRUSTS_STRINGS,
    TRUSTS,
    CORP_WORDS,
    PO_BOXES
)

class CleanStringBase:

    """Functions that accept a single string as an argument and return the string cleaned."""

    # MOVE THESE? aren't necessarily string cleaners
    @classmethod
    def core_name(cls, text: str) -> str:
        """
        Extracts core name from clean name by removing all words found in CORE_NAME_KEYS.

        Examples:
            'Oak Grove Properties' -> 'Oak Grove'
            'Pleasant View Apartments' -> 'Pleasant View'
        """
        try:
            text = text + " "
            for key in CORE_NAME_KEYS:
                text = re.sub(r"{}".format(key), "", text)
            return text.strip()
        except:
            return text

    @classmethod
    def get_is_bank(cls, text: str) -> bool:
        """Returns True if the string contains keywords associated with banks."""
        # todo: figure out best way to check if text is in BANK_VALUES
        if pd.notnull(text):
            for key in BANK_VALUES:
                if key in text:
                    return True
        return False

    @classmethod
    def get_is_trust(cls, text: str) -> bool:
        """Returns True if the string contains keywords associated with trusts."""
        if pd.notnull(text):
            name_split = text.split()
            for name in name_split:
                if name in TRUSTS:
                    return True
            for t in TRUSTS_STRINGS:
                if t in text:
                    return True
        return False

    @classmethod
    def get_is_person(cls, text: str) -> bool:
        """Returns True if the string is identified as a person."""
        # todo: figure out a better way to check if text is in NAMES_LIST
        try:
            text_list = text.split()
            if len(text_list) > 2 and text_list[-1] in ["JR", "SR"]:
                if all(word in NAMES_LIST for word in text_list[:-1]):
                    return True
            if all(word in NAMES_LIST for word in text_list):
                return True
            return False
        except:
            return False

    @classmethod
    def get_is_common_name(cls, text: str) -> bool:
        """Checks common names data and returns True if the name passed as a parameter is found within it."""
        # todo: figure out a better way to check if text is in COMMON_NAMES
        return text in COMMON_NAMES

    @classmethod
    def get_is_org(cls, text: str) -> bool:
        """Returns True if the string contains keywords associated with organizations."""
        try:
            all_words = []
            for item in text.split():
                if item in CORP_WORDS:
                    all_words.append(True)
                else:
                    all_words.append(False)
            if True in all_words:
                return True
            else:
                return False
        except:
            return False

    @classmethod
    def get_is_llc(cls, text: str) -> bool:
        """Returns True if the string contains LLC."""
        text_split = text.split()
        if "LLC" in text_split or "LLC" in text[-4:]:
            return True
        else:
            return False

    @classmethod
    def get_is_pobox(cls, text: str) -> bool:
        """Returns True if a PO box address pattern is identified in the string."""
        addr_spaces_removed = text.replace(" ", "")
        for po in PO_BOXES:
            if addr_spaces_removed.startswith(po):
                return True
        return False

    @classmethod
    def make_upper(cls, text: str) -> str:
        """
        Converts text to uppercase.

        Examples:
            'Smith LLC' -> 'SMITH LLC'
            'Chicago Title' -> 'CHICAGO TITLE'
            'Property Management' -> 'PROPERTY MANAGEMENT'
            'Main St.' -> 'MAIN ST.'
        """
        if pd.isna(text):
            return ""
        return text.upper()

    @classmethod
    def remove_symbols_punctuation(cls, text: str) -> str:
        """
        Removes punctuation and special symbols from text, with special handling for:
        - Converts '&' to 'AND'
        - Preserves '/' and '-' characters
        - Replaces ',' and '.' with spaces
        - Replaces other punctuation with spaces

        Examples:
            'Smith & Sons, LLC.' -> 'SMITH AND SONS LLC'
            'A.B.C. Corp.' -> 'A B C  CORP '
            'Fast-Food/Cafe' -> 'Fast-Food/Cafe'
            'Main St., #401' -> 'Main St  401'
        """
        text = text.replace("&", " AND ")
        text = text.replace(",", " ")
        text = text.replace(".", " ")
        return text.translate(
            str.maketrans(
                string.punctuation.replace("/", "").replace("-", ""),
                " "*len(string.punctuation.replace("/", "").replace("-", ""))
            )
        )

    @classmethod
    def trim_whitespace(cls, text: str) -> str:
        """
        Removes leading and trailing whitespace from text.

        Examples:
            '  HELLO WORLD  ' -> 'HELLO WORLD'
            'ABC   ' -> 'ABC'
            '   XYZ' -> 'XYZ'
            '  SMITH LLC  ' -> 'SMITH LLC'
        """
        return text.strip()

    @classmethod
    def remove_extra_spaces(cls, text: str) -> str:
        """
        Replaces multiple consecutive spaces with a single space.

        Examples:
            'SMITH    LLC' -> 'SMITH LLC'
            'ABC   CORP' -> 'ABC CORP'
            'FIRST    SECOND   THIRD' -> 'FIRST SECOND THIRD'
            'TOO   MANY    SPACES' -> 'TOO MANY SPACES'
        """
        return re.sub(r"\s+", " ", text)

    @classmethod
    def words_to_num(cls, text: str) -> str | int:
        """
        Converts numbers spelled out to integers, handling special case for "POINT" since it might be part of a decimal

        Examples:
            'SEVEN' -> 7
            'NINE' -> 9
        """
        # todo: fix so that it converts SPLIT text, NOT the entire string
        if text == "POINT":
            return text
        else:
            try:
                return w2n.word_to_num(text)
            except:
                return text

    @classmethod
    def deduplicate(cls, text: str) -> str:
        """
        Removes all duplicate words in text, reducing to single instances.

        Examples:
            'SMITH SMITH LLC' -> 'SMITH LLC'
            'THE THE THE CORPORATION' -> 'THE CORPORATION'
            'FIRST FIRST SECOND SECOND TRUST' -> 'FIRST SECOND TRUST'
        """
        text_list = text.split()
        return ' '.join(dict.fromkeys(text_list))

    @classmethod
    def convert_ordinals(cls, text: str) -> str:
        """
        Converts word numbers and ordinals in street names to numeric form while preserving directionals.

        Examples:
            'TENTH STREET' -> '10TH STREET'
            'TENTH STREET EAST' -> '10TH STREET EAST'
            'WEST TENTH STREET' -> 'WEST 10TH STREET'
            'EAST TWENTY FIFTH STREET' -> 'EAST 25TH STREET'
            'FORTY SECOND AVENUE NORTH' -> '42ND AVENUE NORTH'
            'SOUTH ONE HUNDREDTH STREET' -> 'SOUTH 100TH STREET'
        """
        try:
            if (type(cls.words_to_num(text.split("TH")[0])) == int) and (text[-2:] == "TH"):
                return str(cls.words_to_num(text.split("TH")[0])) + "TH"
        except:
            return text

    @classmethod
    def take_first(cls, text):
        """
        Extracts first number from a hyphenated range (e.g., '123-456') and removes the range.

        Examples:
            '123-456 MAIN ST' -> '123 MAIN ST'
            '789-791 OAK AVE' -> '789 OAK AVE'
            '555-559 BUILDING A-1' -> '555 BUILDING A-1'
        """
        try:
            text = re.findall(r"\d+-\d+", text)[0].split("-")[0] + re.sub(r"\d+-\d+", "", text)
        except:
            return text
        return text

    @classmethod
    def combine_numbers(cls, text: str) -> str:
        """
        Combines sequences of numbers in a list while handling special cases with zeros.

        Examples:
            '1 2 3 MAIN' -> '123 MAIN'
            '20 15 OAK' -> '2015 OAK'
            10 5 25 'ST' -> '10525 ST'
            'NO 1 2 WAY' -> 'NO 12 WAY'
        """
        if pd.isna(text):
            return ""
        text_list = text.split()
        whole_list = []
        start = False
        end = False
        text_list.append("JUNK")  # Sentinel value to process final group
        numbers = []

        for p in text_list:
            try:
                int(p)  # Try to convert to integer
                numbers.append(p)
                start = True
            except:
                if start is True:
                    end = True
                else:
                    end = False
                    whole_list.append(p)

            if start and end:
                if len(numbers) == 1:
                    whole_list.append(numbers[0])
                elif len(numbers) == 2:
                    if str(numbers[0])[-1:] == "0" and str(numbers[1])[-1:] != "0":
                        complete = str(str(numbers[0]) + str(numbers[1]))
                    else:
                        complete = str(numbers[0]) + str(numbers[1])
                    whole_list.append(complete)
                else:
                    if str(numbers[0])[-1:] == "0":
                        complete = str(numbers[0][:-1] + str(numbers[1]))
                    else:
                        complete = str(numbers[0]) + str(numbers[1])
                    for i in numbers[2:]:
                        if complete[-1:] == "0":
                            complete = str(str(complete) + str(i))
                        else:
                            complete = complete + str(i)
                    whole_list.append(complete)
                numbers = []
                start = False
                end = False
                whole_list.append(p)

        return " ".join(whole_list[:-1])  # Remove the 'JUNK' sentinel

class CleanStringName(CleanStringBase):

    @classmethod
    def switch_the(cls, text: str) -> str:
        """
        Moves 'THE' from end of string to beginning if it appears at the end.
        Common in property/business names where 'THE' may be misplaced.

        Examples:
            'BUILDING THE' -> 'THE BUILDING'
            'PROPERTY GROUP THE' -> 'THE PROPERTY GROUP'
            'THE CORPORATION' -> 'THE CORPORATION'  (unchanged)
            'SMITH LLC THE' -> 'THE SMITH LLC'
        """
        if text[-4:] == " THE":
            return "THE " + text[:-4]
        else:
            return text


class CleanStringAddress(CleanStringBase):

    @classmethod
    def convert_nsew(cls, text: str) -> str:
        """
        Converts cardinal directions to abbreviated forms, but only converts the first direction
        when two directions appear consecutively and are immediately followed by a street suffix.
        Otherwise, leaves them unchanged.

        Examples:
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
                        if i + 2 < len(words) and words[i + 2] in STREET_SUFFIXES:
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


    @classmethod
    def remove_secondary_designators(cls, text: str) -> str:
        """
        Removes unit designations from addresses, while retaining the unit number.

        Examples:
            '123 MAIN ST #4B' -> '123 MAIN ST 4B'
            '456 OAK AVE APT 7' -> '456 OAK AVE 7'
            '789 PINE LN UNIT 2C' -> '789 PINE LN 2C'
            '321 MAPLE DR SUITE 5' -> '321 MAPLE DR 5'
        """
        text_split = text.split()
        try:
            if text_split[0].isdigit() and text_split[1] in SECONDARY_KEYWORDS:
                return text
            for keyword in SECONDARY_KEYWORDS:
                if keyword in text_split:
                    index = text_split.index(keyword)
                    text_split.pop(index)  # Remove the keyword itself
                    break
            return " ".join(text_split)
        except:
            return text

    @classmethod
    def convert_street_suffixes(cls, text: str) -> str:
        """
        Replaces full street suffix names with their standardized abbreviations using the STREET_SUFFIXES mapping.

        Examples:
        '123 MAIN STREET' -> '123 MAIN ST'
        '456 OAK AVENUE' -> '456 OAK AVE'
        '789 PINE BOULEVARD' -> '789 PINE BLVD'
        '321 MAPLE DRIVE' -> '321 MAPLE DR'
        """
        words = text.split()
        for i, word in enumerate(words):
            if word in STREET_SUFFIXES.keys():
                words[i] = STREET_SUFFIXES[word]
        return " ".join(words)


    @classmethod
    def fix_zip(cls, text):
        """
        Normalizes ZIP codes to a five-digit format, removing non-numeric characters and padding with leading zeros if necessary.

        Examples:
            '123' -> '00123'
            '45678' -> '45678'
            '9876' -> '09876'
            '12A34' -> ''
        """
        if text.isdigit():
            if len(text) == 5:
                return text
            elif len(text) == 9:
                return f"{text[:5]}-{text[5:]}"
            elif len(text) > 5:
                return text[:5]
            else:
                return str(int("".join(filter(str.isdigit, text)))).zfill(5)
        else:
            return ""

    @classmethod
    def check_sec_num(cls, text: str) -> str | float:
        """
        Sting passed into `text` must be fully formatted address. Searches for numbers at the end of street addresses
        to check for missing secondary numbers in the validated addresses.
        """
        street: str = text.split(",")[0].strip()
        match = re.search(r"(\d+)$", street)
        if match:
            return match.group(1)
        else:
            return np.nan



class CleanStringAccuracy(CleanStringBase):

    """String cleaning functions that could meaningfully impact accuracy during matching processes."""

    @classmethod
    def drop_letters(cls, text):
        """
        Removes single letters that follow numbers (e.g., '123A' -> '123').
        Useful for standardizing addresses where unit letters might be inconsistent.

        Examples:
            '123A MAIN ST' -> '123 MAIN ST'
            '456B OAK AVE' -> '456 OAK AVE'
            '789C BUILDING' -> '789 BUILDING'
            'APT 101D' -> 'APT 101'
        """
        try:
            text = re.sub(r"\d+[a-zA-Z]", re.findall(r"\d+[a-zA-Z]", text)[0][:-1], text)
        except:
            return text
        return text

    @classmethod
    def convert_mixed(cls, text):
        """
        Converts word numbers to digits while preserving surrounding text.

        Examples:
            '2 TWENTY MAIN ST' -> '2 20 MAIN ST'
            '55 THIRTY AVE' -> '55 30 AVE'
            '1 HUNDRED STREET' -> '1 100 STREET'
            'APT FORTY 2' -> 'APT 40 2'
        """
        try:
            # Split into words
            words = text.split()
            converted_words = []

            for word in words:
                # If word has digits, keep as is
                if any(c.isdigit() for c in word):
                    converted_words.append(word)
                    continue

                # Try to convert word number
                try:
                    num = cls.words_to_num(word)
                    if isinstance(num, int):
                        converted_words.append(str(num))
                    else:
                        converted_words.append(word)
                except:
                    converted_words.append(word)

            return ' '.join(converted_words)
        except:
            return text

    @classmethod
    def remove_secondary_component(cls, text: str) -> str:
        """
        Removes entire secondary component from the address. Does NOT retain unit number.

        Examples:
            '123 MAIN ST #4B' -> '123 MAIN ST'
            '456 OAK AVE APT 7' -> '456 OAK AVE'
            '789 PINE LN UNIT 2C' -> '789 PINE LN'
            '321 MAPLE DR SUITE 5' -> '321 MAPLE DR'
        """
        try:
            for keyword in SECONDARY_KEYWORDS:
                text = text.split(keyword)[0].strip()
            return text
        except:
            return text
