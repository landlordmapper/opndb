class SummaryStats(object):
    # todo: create excel file with tabs for different workflow stages and save summary stats in each tab
    @classmethod
    def summary_stats_data_load(cls):
        # unique property count
        # properties with missing names/addresses counts
        # unique corp count
        # corps with missing names/addresses counts
        # unique LLC count
        # LLCs with missing names/addresses counts
        pass

    @classmethod
    def summary_stats_data_clean(cls):
        pass

    @classmethod
    def summary_stats_address_initial(cls):
        # unique raw address count total
        # unique raw taxpayer address count
        # unique raw corp & LLC count, total AND per column (differentiate by manager/member addr, office addr, etc.)
        # po box addrs detected
        # po box addrs successfully validated
        # total validated & unvalidated after initial processing
        pass

    @classmethod
    def summary_stats_address_open_addrs(cls):
        # number of addresses successfully validated by open addresses workflow
        # total validated & unvalidated after open addrs processing
        pass

    @classmethod
    def summary_stats_address_geocodio(cls):
        # number of addresses passed into geocodio
        # number of addresses successfully validated by geocodio
        # number of addresses unsuccessfully validated by geocodio
        # number of failed geocodio calls
        # final total validated & unvalidated addresses after all validation processing
        pass

    @classmethod
    def summary_stats_name_analysis(cls):
        # list of standardized names and counts of different variations that were standardized
        # ex:
        # CHICAGO LAND TRUST, 473
        # US BANK, 253
        # DEVON TRUST, 117
        pass

    @classmethod
    def summary_stats_address_analysis(cls):
        # counts for ll_orgs, law firms, etc
        pass

    @classmethod
    def summary_stats_rental_subset(cls):
        # number of total properties in dataset
        # number of properties in initial subset
        # number of non-rental properties pulled in from taxpayer addrs after subsetting
        pass

    @classmethod
    def summary_stats_clean_merge(cls):
        # number of taxpayer records associated with corps
        # number of taxpayer records associated with llcs
        # number of unique corps pulled in
        # number of unique LLCs pulled in
        # number of taxpayer records identified as corp/entities but NOT assigned a corp/LLC
        # number of IS_LLC == True rows vs number of matched LLCs
        pass

    @classmethod
    def summary_stats_string_matching(cls):
        # use stats from methodology paper
        pass

    @classmethod
    def summary_stats_network_graph(cls):
        # use stats from methodology paper
        pass

    @classmethod
    def summary_stats_final_output(cls):
        # final dataset stats
        pass
