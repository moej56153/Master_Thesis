from astroquery.heasarc import Heasarc, Conf
import astropy.units as u
from astropy.coordinates import SkyCoord
import pandas as pd 

 class integral_query:
     def __init__(self, mission="integral_rev3_scw", object_name=None, coords=None, radius=None, sortvar="START_DATE"):
         pass