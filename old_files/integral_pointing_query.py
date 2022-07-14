from astroquery.heasarc import Heasarc, Conf
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd 

heasarc = Heasarc()
Conf.server.set('https://www.isdc.unige.ch/browse/w3query.pl')


class integral_query:
    def __init__(self, 
            object_name = None, 
            position = None, # Example: "SkyCoord('12h29m06.70s +02d03m08.7s', frame='icrs')" 
            radius = None, # Example: "120 arcmin" or "120*u.arcmin"
            mission = "integral_rev3_scw", 
            sortvar = "START_DATE", # _SEARCH_OFFSET doens't seem to work, use sort_by("_SEARCH_OFFSET") method instead
            resultsmax = 0
            ):
        
        self.query_dict={}
        
        if not object_name is None: self.query_dict["object_name"] = object_name
        if not position is None: self.query_dict["position"] = position
        if not radius is None: self.query_dict["radius"] = radius
        if not mission is None: self.query_dict["mission"] = mission
        if not sortvar is None: self.query_dict["sortvar"] = sortvar
        if not resultsmax is None: self.query_dict["resultsmax"] = resultsmax


        if not object_name is None and position is None:
            self.table = heasarc.query_object(**self.query_dict)

        elif object_name is None and not position is None and not radius is None:
            self.table = heasarc.query_region(**self.query_dict)

        else:
            raise TypeError("Please specify either object_name, or position and radius")

        self.table.convert_bytestring_to_unicode()
        self.table = self.table.to_pandas()

        int_cols = ["SCW_ID", "SCW_VER"]
        self.table[int_cols] = pd.to_numeric(self.table[int_cols].stack()).unstack()

        flt_cols = ["RA_X", "DEC_X", "_SEARCH_OFFSET", "OBS_ID", "GOOD_SPI", "GOOD_PICSIT", "GOOD_ISGRI", "GOOD_JEMX", "GOOD_JEMX1", "GOOD_JEMX2", "GOOD_OMC", "DSIZE"]
        self.table[flt_cols] = pd.to_numeric(self.table[flt_cols].stack(), errors="coerce").unstack()

        dat_cols = ["START_DATE", "END_DATE"]
        self.table[dat_cols] = pd.to_datetime(self.table[dat_cols].stack()).unstack()

        str_cols = ["SCW_TYPE", "OBS_TYPE", "PS", "PI_NAME"]
        self.table[str_cols] = self.table[str_cols].stack().str.strip().unstack()

        self.filtered_table = self.table.copy()
        self.SCW_ID_list = self.filtered_table["SCW_ID"]

    def sort_by(self, sortvar):
        self.table = self.table.sort_values(sortvar)
        self.filtered_table = self.filtered_table.sort_values(sortvar)
        self.SCW_ID_list = self.filtered_table["SCW_ID"]
    
    def reset_filter(self):
        self.filtered_table = self.table.copy()
        self.SCW_ID_list = self.filtered_table["SCW_ID"]

    def filter(self,
        SCW_ID = None, # Match Value, int
        SCW_VER = None, # Match Value, int
        SCW_TYPE = None, # Match Value, str
        RA_X = (None,None), # (lower_limit, upper_limit), flt
        DEC_X = (None,None), # (lower_limit, upper_limit), flt
        START_DATE = (None,None), # (lower_limit, upper_limit), str
        END_DATE = (None,None), # (lower_limit, upper_limit), str
        OBS_ID = None, # Match Value, flt
        OBS_TYPE = None, # Match Value, str
        PS = None, # Match Value, str
        PI_NAME = None, # Match Value, str
        GOOD_SPI = (None,None), # (lower_limit, upper_limit), flt
        GOOD_PICSIT = (None,None), # (lower_limit, upper_limit), flt
        GOOD_JMEX = (None,None), # (lower_limit, upper_limit), flt
        GOOD_JMEX1 = (None,None), # (lower_limit, upper_limit), flt
        GOOD_JMEX2 = (None,None), # (lower_limit, upper_limit), flt
        GOOD_OMC = (None,None), # (lower_limit, upper_limit), flt
        DSIZE = (None,None), # (lower_limit, upper_limit), flt
        _SEARCH_OFFSET = (None,None), # (lower_limit, upper_limit), flt
        ):

        if not SCW_ID is None: self.filtered_table = self.filtered_table[self.filtered_table["SCW_ID"] == SCW_ID]

        if not SCW_VER is None: self.filtered_table = self.filtered_table[self.filtered_table["SCW_VER"] == SCW_VER]

        if not SCW_TYPE is None: self.filtered_table = self.filtered_table[self.filtered_table["SCW_TYPE"] == SCW_TYPE.strip().upper()]

        if not RA_X[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["RA_X"] >= RA_X[0]]
        if not RA_X[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["RA_X"] <= RA_X[1]]

        if not DEC_X[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["DEC_X"] >= DEC_X[0]]
        if not DEC_X[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["DEC_X"] <= DEC_X[1]]

        if not START_DATE[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["START_DATE"] >= pd.to_datetime(START_DATE[0])]
        if not START_DATE[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["START_DATE"] <= pd.to_datetime(START_DATE[1])]

        if not END_DATE[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["END_DATE"] >= pd.to_datetime(END_DATE[0])]
        if not END_DATE[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["END_DATE"] <= pd.to_datetime(END_DATE[1])]

        if not OBS_ID is None: self.filtered_table = self.filtered_table[self.filtered_table["OBS_ID"] == float(OBS_ID)]

        if not OBS_TYPE is None: self.filtered_table = self.filtered_table[self.filtered_table["OBS_TYPE"] == OBS_TYPE.strip().upper()]

        if not PS is None: self.filtered_table = self.filtered_table[self.filtered_table["PS"] == PS.strip().upper()]

        if not PI_NAME is None: self.filtered_table = self.filtered_table[self.filtered_table["PI_NAME"] == PI_NAME.strip().upper()]

        if not GOOD_SPI[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_SPI"] >= GOOD_SPI[0]]
        if not GOOD_SPI[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_SPI"] <= GOOD_SPI[1]]

        if not GOOD_PICSIT[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_PICSIT"] >= GOOD_PICSIT[0]]
        if not GOOD_PICSIT[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_PICSIT"] <= GOOD_PICSIT[1]]

        if not GOOD_JMEX[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_JMEX"] >= GOOD_JMEX[0]]
        if not GOOD_JMEX[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_JMEX"] <= GOOD_JMEX[1]]

        if not GOOD_JMEX1[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_JMEX1"] >= GOOD_JMEX1[0]]
        if not GOOD_JMEX1[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_JMEX1"] <= GOOD_JMEX1[1]]

        if not GOOD_JMEX2[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_JMEX2"] >= GOOD_JMEX2[0]]
        if not GOOD_JMEX2[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_JMEX2"] <= GOOD_JMEX2[1]]

        if not GOOD_OMC[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_OMC"] >= GOOD_OMC[0]]
        if not GOOD_OMC[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["GOOD_OMC"] <= GOOD_OMC[1]]

        if not DSIZE[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["DSIZE"] >= DSIZE[0]]
        if not DSIZE[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["DSIZE"] <= DSIZE[1]]

        if not _SEARCH_OFFSET[0] is None: self.filtered_table = self.filtered_table[self.filtered_table["_SEARCH_OFFSET"] >= _SEARCH_OFFSET[0]]
        if not _SEARCH_OFFSET[1] is None: self.filtered_table = self.filtered_table[self.filtered_table["_SEARCH_OFFSET"] <= _SEARCH_OFFSET[1]]

        self.SCW_ID_list = self.filtered_table["SCW_ID"]
        return self.SCW_ID_list




# Example Usage

# Make query for object
query1 = integral_query(object_name="Cyg X-1")
# Change sorting parameter
query1.sort_by("_SEARCH_OFFSET")
# Retrieve SCW_ID list for some filters
Q1L1 = query1.filter(START_DATE=("2020-07-01 00:00:00","2021-07-01 00:00:00"), DSIZE=(100, None), SCW_TYPE="  pointing    ")
# Retrieve SCW_ID list for additional filters
Q1L2 = query1.filter(_SEARCH_OFFSET=(None, 30))
# Reset filters
query1.reset_filter()
# Retrieve SCW_ID list for some other filters
Q1L3 = query1.filter(_SEARCH_OFFSET=(None, 10))

# Query for space coordinates
query2 = integral_query(position=SkyCoord('12h29m06.70s +02d03m08.7s', frame='icrs'), radius="3 degree")
# Retrieve SCW_ID list
Q2L1 = query2.filter()


