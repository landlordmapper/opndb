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