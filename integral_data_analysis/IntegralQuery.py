from dataclasses import dataclass, asdict 
from asciitree import LeftAligned
from asciitree.drawing import BOX_DOUBLE, BoxStyle
from typing import Union
from datetime import datetime
import warnings 
from astroquery.heasarc import Heasarc, Conf
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd 
import numpy as np

heasarc = Heasarc()
Conf.server.set('https://www.isdc.unige.ch/browse/w3query.pl')

def _to_print_dict(conf):
    out = {}

    if not isinstance(conf, dict):

        dummy = {}
        dummy[f"{conf}"] = {}

        return dummy
    
    else:
        for k, v in conf.items():

            text = f"{k}"

            out[text] = _to_print_dict(v)

    return out


class BaseDataClass:
    def show(self):
        final = {}
        final[type(self).__name__] = _to_print_dict(asdict(self))
        tr = LeftAligned(draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))
        print(tr(final))

        
@dataclass
class Range(BaseDataClass):
    """
    Simple dataclass that represents a range
    """
    min_val: Union[str, float, None] = None 
    max_val: Union[str, float, None] = None 
    
    
@dataclass
class Filter(BaseDataClass):
    """
    Dataclass to handle the filter parameters
    """
    SCW_ID: Union[str, None] = None 
    SCW_VER: Union[int, None] = None 
    SCW_TYPE: Union[int, None] = None 
    RA_X: Union[Range, None] = None
    DEC_X: Union[Range, None] = None
    TIME: Union[Range, None] = None
    #END_DATE: Union[Range, None] = None
    OBS_ID: Union[str, None] = None
    OBS_TYPE: Union[str, None]  = None
    PS: Union[str, None]  = None
    PI_NAME: Union[str, None]  = None
    GOOD_SPI: Union[Range, None]  = None
    GOOD_PICSIT: Union[Range, None] = None
    GOOD_JMEX: Union[Range, None] = None
    GOOD_JMEX1: Union[Range, None] = None
    GOOD_JMEX2: Union[Range, None] = None
    GOOD_OMC: Union[Range, None] = None
    DSIZE: Union[Range, None] = None
    _SEARCH_OFFSET: Union[Range, None] = None
    
    
@dataclass
class SearchQuery(BaseDataClass):
    """
    Dataclass to handle the inital search query parameters
    """
    object_name: Union[str, None] = None
    position: Union[SkyCoord, None] = None
    radius: Union[float, str, None] = None
    mission: Union[str, None] = "integral_rev3_scw"
    sortvar: Union[str, None] = "START_DATE"
    resultmax: Union[int, None] = 0
    
    @property
    def object_dict(self):
        """
        Get dict for the object modus => drop radius and position
        :returns:
        """
        dic = asdict(self)
        dic.pop("radius")
        dic.pop("position")
        return dic
    
    @property
    def region_dict(self):
        """
        Get dict for the region modus => drop object_name
        :returns:
        """
        dic = asdict(self)
        dic.pop("object_name")
        return dic
    

class IntegralQuery:
    def __init__(self, search_query: SearchQuery):
        """
        Init the Integral query object. Used to get the SCW_ID for a certain position or
        object and apply different filters to it
        """
        assert (search_query.object_name is not None or 
                (search_query.position is not None and 
                 search_query.radius is not None)), "Please specify either object_name, or position and radius"
        
        if search_query.object_name:
            self._table = heasarc.query_object(**search_query.object_dict)

        else:
            self._table = heasarc.query_region(**search_query.region_dict)
            
        self._format_table()

    def _format_table(self):
        """
        Format the table
        :returns:
        """
        # BB: self._table
        self._table.convert_bytestring_to_unicode()
        self._table = self._table.to_pandas()

        int_columns = ["SCW_VER"]
        float_columns = ["RA_X", "DEC_X", "GOOD_SPI", "GOOD_PICSIT", "GOOD_ISGRI", 
                         "GOOD_JEMX", "GOOD_JEMX1", "GOOD_JEMX2", "GOOD_OMC", "DSIZE"]
        string_columns = ["SCW_ID", "SCW_TYPE", "OBS_TYPE","OBS_ID", "PS", "PI_NAME"]
        datetime_columns = ["START_DATE", "END_DATE"]
        
        for c in int_columns:
            self._table[c] = self._table[c].astype(int)
            
        for c in float_columns:
            self._table[c] = self._table[c].str.strip()
            mask = self._table.copy()[c]==""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._table[c].loc[mask] = 0
            self._table[c] = self._table[c].astype(float)
            
        for c in string_columns:
            self._table[c] = self._table[c].str.strip()
            
        for c in datetime_columns:
            self._table[c] = pd.to_datetime(self._table[c])

    def sort_by(self, sortvar):
        """
        Sort the tables by a new Variable
        :param sortvar: Variable to use for the sort
        :returns:
        """
        self._table = self._table.sort_values(sortvar)

    def apply_filter(self, filter_ob: Filter, return_coordinates=False, remove_duplicates=True) -> np.array:
        """
        Apply a filter to the base table
        :param filter_ob: Filter Object with all the filter parameters
        :param return_coordintes: Specifies if coordintes should be returned
        :returns:
        """
        new_table = self._table.copy()
        for key, value in asdict(filter_ob).items():
            if value:
                if type(value) is dict:
                    if key=="TIME":
                        if value["min_val"]:
                            new_table = new_table[new_table["START_DATE"]>=
                                                  datetime.fromisoformat(value["min_val"])]
                        if value["max_val"]:
                            new_table = new_table[new_table["END_DATE"]<=
                                                  datetime.fromisoformat(value["max_val"])]
                    else:
                        if value["min_val"]:
                            new_table = new_table[new_table[key]>=value["min_val"]]
                        if value["max_val"]:
                            new_table = new_table[new_table[key]<=value["max_val"]]
                else:
                    new_table = new_table[new_table[key]==value]
                    
        if remove_duplicates:
            new_table = new_table.drop_duplicates("SCW_ID", keep="last")
        
        if not return_coordinates:
            return new_table["SCW_ID"].to_numpy()
        else:
            return np.concatenate((new_table[["SCW_ID","RA_X","DEC_X"]].to_numpy(), 
                                   np.array([new_table["START_DATE"].dt.to_pydatetime()]).T), axis=1)
    
    @property
    def table(self):
        """
        :returns: Base Table of Query
        """
        return self._table