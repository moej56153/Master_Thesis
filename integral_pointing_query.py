from astroquery.heasarc import Heasarc, Conf
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd 

integral_query = Heasarc()
Conf.server.set('https://www.isdc.unige.ch/browse/w3query.pl')

class integral_query:
    def __init__(self, mission="integral_rev3_scw", object_name=None, coords=None, radius=None, sortvar="START_DATE", resultsmax=0):
        self.query_dict={}
        if not object_name is None: self.query_dict{object_name}=object_name
        if not coords is None: self.query_dict{coords}=coords
        if not radius is None:resultmax self.query_dict{radius}=radius
        if not resultmax is None: self.query_dict{resultmax}=resultmax


