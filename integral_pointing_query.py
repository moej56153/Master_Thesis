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
            resultsmax = 0):
        
        self.query_dict={}
        
        if not object_name is None: self.query_dict["object_name"] = object_name
        if not position is None: self.query_dict["position"] = position
        if not radius is None: self.query_dict["radius"] = radius
        if not mission is None: self.query_dict["mission"] = mission
        if not sortvar is None: self.query_dict["sortvar"] = sortvar
        if not resultsmax is None: self.query_dict["resultsmax"] = resultsmax


        if not object_name is None and position is None:
            r = heasarc.query_object(**self.query_dict)

        elif object_name is None and not position is None and not radius is None:
            r = heasarc.query_region(**self.query_dict)

        else:
            raise TypeError("Please specify either object_name, or position and radius")

        r.convert_bytestring_to_unicode()
        p = r.to_pandas()

        
#test2 = integral_query(object_name="Cyg X-1")


test = integral_query(position=SkyCoord('12h29m06.70s +02d03m08.7s', frame='icrs'), radius="1 degree")


