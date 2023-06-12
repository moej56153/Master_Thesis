import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from astromodels import Powerlaw, Line, PointSource, SpectralComponent
import astropy.time as at
from datetime import datetime
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
from MultinestClusterFit import powerlaw_binned_spectrum
import pickle
from RebinningFunctions import *


data_path = "./main_files/SPI_data/0374"

source_ra, source_dec = 10, -40
source_slope = 8e-6

pointing_index = 1

data_path_d = "./main_files/energy_range_error_test/bkg_20_lin_source_8__6_data"

if not os.path.exists(f"{data_path_d}"):
    os.mkdir(f"{data_path_d}")
    
with open(f"{data_path_d}/source_params.pickle", "wb") as f:
    pickle.dump((source_ra, source_dec, source_slope), f)


# Energy Bins
with fits.open(f"{data_path}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
# Pointings and Start Times
with fits.open(f"{data_path}/pointing.fits") as file:
    t = Table.read(file[1])
    
    pointings = np.array(t["PTID_SPI"])
    
    time_start = np.array(t["TSTART"]) + 2451544.5
    time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
    time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
    
# Time Elapsed
# det=i, pointing_index=j : index = j*85 + i
with fits.open(f"{data_path}/dead_time.fits") as file:
    t = Table.read(file[1])
    time_elapsed = np.array(t["LIVETIME"])
    
time_elapsed = np.full(time_elapsed.shape, 3000.)
    
updated_time = t.copy()
updated_time["LIVETIME"] = time_elapsed


# Only necessary for 1380
skip_pointing = [False] * len(pointings)
# skip_pointing[0] = True

# Background

with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
    t = Table.read(file[1])
    
background_counts = t.copy()

background_counts["COUNTS"] = np.full(background_counts["COUNTS"].shape, 20.)
        
background_counts["COUNTS"] = np.random.poisson(background_counts["COUNTS"])

assert find_response_version(time_start[0]) == find_response_version(time_start[-1]), "Versions not constant"
version = find_response_version(time_start[0])
rsp_base = ResponseDataRMF.from_version(version)


resp_mats = []
emod = np.geomspace(10, 3000, 50)

for p_i, pointing in enumerate(pointings):
    if skip_pointing[p_i]:
        continue
    
    time = time_start[p_i]
    dets = get_live_dets(time=time, event_types=["single"])
    
    rmfs = []
    for d in dets:
        rmfs.append(ResponseRMFGenerator.from_time(time, d, energy_bins, emod, rsp_base))
        
    sds = np.empty(0)
    for d in range(len(dets)):
        sd = SPIDRM(rmfs[d], source_ra, source_dec)
        sds = np.append(sds, sd.matrix.T)
    resp_mats.append(sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1)))
    
def calc_count_rates(resp_mats, ra, dec, b):
    s = Line()
    s.a = 0
    s.a.free = False
    s.b = b
    component1 = SpectralComponent("line", shape=s)
    source = PointSource("Test", ra=ra, dec=dec, components=[component1])
    
    spec = source(emod)
    spec_binned = (emod[1:]-emod[:-1])*(spec[:-1]+spec[1:])/2
    
    source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1), dtype=np.uint32)
    
    for p_i, pointing in enumerate(pointings):
        if skip_pointing[p_i]:
            continue
        
        resp_mat = resp_mats[p_i]
        
        count_rates = np.dot(spec_binned, resp_mat)
        
        for d_i, d in enumerate(dets):
            index = p_i * 85 + d
            source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])
    
    return source_counts

temp_path = f"{data_path_d}"        

if not os.path.exists(temp_path):
    os.mkdir(temp_path)
    
os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

        
hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
hdu.writeto(f"{temp_path}/dead_time.fits")

source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_slope)

total_counts = background_counts.copy()
total_counts["COUNTS"] += source_counts
        
hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
hdu.writeto(f"{temp_path}/evts_det_spec.fits")