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
            sortvar = "START_DATE", 
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

        self.filterd_table = self.table.copy()
        self.SCW_ID_list = self.filterd_table["SCW_ID"]
    
    def reset_filter(self):
        self.filterd_table = self.table.copy()
        self.SCW_ID_list = self.filterd_table["SCW_ID"]

    def filter(self,
        SCW_ID = None, # Match Value
        SCW_VER = None, # Match Value
        SCW_TYPE = None, # Match Value
        RA_X = None, # (lower_limit, upper_limit)
        DEC_X = None, # (lower_limit, upper_limit)
        START_DATE = None, # (lower_limit, upper_limit)
        END_DATE = None, # (lower_limit, upper_limit)
        OBS_ID = None, # Match Value
        OBS_TYPE = None, # Match Value
        PS = None, # Match Value
        PI_NAME = None, # Match Value
        GOOD_SPI = None, # (lower_limit, upper_limit)
        GOOD_PICSIT = None, # (lower_limit, upper_limit)
        GOOD_JMEX = None, # (lower_limit, upper_limit)
        GOOD_JMEX1 = None, # (lower_limit, upper_limit)
        GOOD_JMEX2 = None, # (lower_limit, upper_limit)
        GOOD_OMC = None, # (lower_limit, upper_limit)
        DSIZE = None, # (lower_limit, upper_limit)
        _SEARCH_OFFSET = None, # (lower_limit, upper_limit)
        ):

        self.SCW_ID_list = self.filterd_table["SCW_ID"]
        return self.SCW_ID_list


        
test2 = integral_query(object_name="Cyg X-1", radius="1 degree")


test = integral_query(position=SkyCoord('12h29m06.70s +02d03m08.7s', frame='icrs'), radius="1 degree")


