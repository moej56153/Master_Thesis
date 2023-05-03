from astropy.coordinates import SkyCoord
import numpy as np
from IntegralQuery import SearchQuery, IntegralQuery, Filter, Range
from IntegralPointingClustering import ClusteredQuery
import astropy.io.fits as fits
from astropy.table import Table
from datetime import datetime
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.livedets import get_live_dets
import os
import astropy.time as at
import pickle

rsp_bases = tuple([ResponseDataRMF.from_version(i) for i in range(5)])

def save_clusters(pointings, folder):
    if not os.path.exists(f"./{folder}"):
        os.mkdir(folder)
    with open(f"./{folder}/pointings.pickle", "wb") as f:
        pickle.dump(pointings, f)
        
def load_clusters(folder):
    with open(f"./{folder}/pointings.pickle", "rb") as f:
        pointings = pickle.load(f)
    return pointings

def extract_date_range(path):
    with fits.open(f"{path}/pointing.fits") as file:
        t = Table.read(file[1])
        t1 = at.Time(f'{t["TSTART"][0]+2451544.5}', format='jd')
        t1.format = "isot"
        t2 = at.Time(f'{t["TSTOP"][-1]+2451544.5}', format='jd')
        t2.format = "isot"
    return t1.value, t2.value

class PointingClusters: #add min time diff
    def __init__(
        self,
        orbit_paths,
        min_angle_dif,
        max_angle_dif,
        max_time_dif,
        radius_around_source,
        min_time_elapsed,
        cluster_size_range,
        random_angle_dif_range=None,
        center_ra=83.6333,
        center_dec=22.0144,
    ):
        self._orbit_paths = orbit_paths
        self._min_angle_dif = min_angle_dif
        self._max_angle_dif = max_angle_dif
        self._max_time_dif = max_time_dif
        self._radius_around_source = radius_around_source
        self._min_time_elapsed = min_time_elapsed
        self._cluster_size_range = cluster_size_range
        self._random_angle_dif_range = random_angle_dif_range
        
        pointings = []
        self._get_scw_ids(center_ra, center_dec)
        cq = ClusteredQuery(
            self._scw_ids,
            angle_weight=0.,
            time_weight=1./self._max_time_dif,
            max_distance=1.,
            min_ang_distance=self._min_angle_dif,
            max_ang_distance=self._max_angle_dif,
            cluster_size_range = self._cluster_size_range,
            failed_improvements_max=3,
            suboptimal_cluster_size=min(max(1, self._cluster_size_range[0]), self._cluster_size_range[1]-1),
            close_suboptimal_cluster_size=min(max(1, self._cluster_size_range[0]), self._cluster_size_range[1]-1)
        ).get_clustered_scw_ids()
                
        for size in range(self._cluster_size_range[0], self._cluster_size_range[1] + 1):
            for cluster in cq[size]:
                pointings_cluster = []
                for pointing in cluster:
                    for i, scw in enumerate(self._scw_ids):
                        if scw[0] == pointing:
                            pointings_cluster.append((pointing, self._scw_paths[i]))
                            break
                pointings.append(tuple(pointings_cluster))
                
        self.pointings = tuple(pointings)             
    
    def _get_scw_ids(self, center_ra, center_dec, print_results=False):
        p = SkyCoord(center_ra, center_dec, frame="icrs", unit="deg")
        searchquerry = SearchQuery(position=p, radius=f"{self._radius_around_source} degree",)
        cat = IntegralQuery(searchquerry)

        scw_ids_all = None
        for path in self._orbit_paths:
            f = Filter(
                SCW_TYPE="POINTING",
                TIME=Range(*extract_date_range(path))
            )
            if scw_ids_all:
                scw_ids_all = np.append(
                    scw_ids_all,
                    cat.apply_filter(f, return_coordinates=True, remove_duplicates=True),
                    axis=0
                )
            else:
                scw_ids_all = cat.apply_filter(f, return_coordinates=True, remove_duplicates=True)
                
        scw_ids = []
        paths = []
        
        multiple_files = []
        no_files = []
        no_pyspi = []
        
        num_dets = 19
        eb = np.geomspace(18, 2000, 5)
        emod = np.geomspace(18, 2000, 5)
        
        for i, scw_id in enumerate(scw_ids_all):
            # Query may structure scws differently to our data, we only care about our data
            repeat = False
            for scw_id_2 in scw_ids_all[:i]:
                if scw_id[0][:8] == scw_id_2[0][:8]:
                    repeat = True
                    break
            if repeat:
                continue
            
            good = True
            found = False
            for path in self._orbit_paths:
                with fits.open(f"{path}/pointing.fits") as file:
                    t = Table.read(file[1])
                    index = np.argwhere(t["PTID_ISOC"]==scw_id[0][:8])
                
                if len(index) < 1:
                    no_files.append(scw_id)
                    continue
                                    
                elif len(index) > 1:
                    multiple_files.append(scw_id)
                    good = False
                    
                found = True
                                                
                pointing_info = t[index[-1][0]]
            
                t1 = at.Time(f'{pointing_info["TSTART"]+2451544.5}', format='jd').datetime
                time_start = datetime.strftime(t1,'%y%m%d %H%M%S')
                                
                with fits.open(f"{path}/dead_time.fits") as file2:
                    t2 = Table.read(file2[1])
                    
                    time_elapsed = np.zeros(num_dets)
                    
                    for i in range(num_dets):
                        for j in index:
                            time_elapsed[i] += t2["LIVETIME"][j[0]*85 + i] # this is only true when combining fragmented scws
                                
                dets = get_live_dets(time=time_start, event_types=["single"])
                                
                if not np.amin(time_elapsed[dets]) > self._min_time_elapsed:
                    good = False
                        
                try: # investigate why this is necessary
                    version = find_response_version(time_start)
                    rsp = ResponseRMFGenerator.from_time(time_start, dets[0], eb, emod, rsp_bases[version])
                except:
                    no_pyspi.append(scw_id)
                    good = False
                    
                if good:
                    paths.append(path)
                
            if not found:
                good = False
                
            if good:
                scw_ids.append(scw_id)
                    
        if print_results:
            print("Multiple Files:")
            print(multiple_files)
            print("No Files:")
            print(no_files)
            print("No PySpi:")
            print(no_pyspi)
            print("Good:")
            print(scw_ids)
            
        assert len(paths) == len(scw_ids), "At least one SCW is found in multiple paths"
        
        self._scw_ids = np.array(scw_ids)
        self._scw_paths = paths