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


data_path = "./main_files/SPI_data/0374"
source_ra, source_dec = 10, -40
source_piv = 200.
source_K = 1e-4
source_index = -2
pointing_index = 1

data_path_d = "./main_files/ppc_test_new"

with open(f"{data_path_d}/source_params.pickle", "wb") as f:
    pickle.dump((source_ra, source_dec, source_piv, source_K, source_index), f)

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
    
updated_time = t.copy()
    
for i in range(int(len(time_elapsed) / 85)):
    if i == pointing_index:
        continue
    else:
        updated_time[i*85 : (i+1)*85] = updated_time[pointing_index*85 : (pointing_index+1)*85]
# Only necessary for 1380
skip_pointing = [False] * len(pointings)
# skip_pointing[0] = True
# Background

with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
    t = Table.read(file[1])
    
background_counts = t.copy()

for i in range(int(len(background_counts) / 85)):
    if i == pointing_index:
        continue
    else:
        background_counts[i*85 : (i+1)*85] = background_counts[pointing_index*85 : (pointing_index+1)*85]
        
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
def calc_count_rates(resp_mats, ra, dec, piv, K, index):
    pl = Powerlaw()
    pl.piv = piv
    pl.K = K
    pl.index = index
    component1 = SpectralComponent("pl", shape=pl)
    source = PointSource("Test", ra=ra, dec=dec, components=[component1])
    
    spec = source(emod)
    spec_binned = powerlaw_binned_spectrum(emod, spec)
    
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


source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_piv, source_K, source_index)

total_counts = background_counts.copy()
total_counts["COUNTS"] += source_counts
total_counts = np.array(total_counts["COUNTS"])

print(total_counts)
print(total_counts.shape)

def pointing_indices_table(data_path, pointings):
    with fits.open(f"{data_path}/pointing.fits") as file:
        t = Table.read(file[1])
        data_pointings = np.array(t["PTID_SPI"])
        time_start = np.array(t["TSTART"]) + 2451544.5
        time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
        time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
        
    assert find_response_version(time_start[0]) == find_response_version(time_start[-1]), "Versions not constant"
    version = find_response_version(time_start[0])
    dets = get_live_dets(time=time_start[0], event_types=["single"])
        
    r_indices = []
    for c_i, cluster in enumerate(pointings):
        t = []
        for p_i, pointing in enumerate(cluster):
            for p_i2, pointing2 in enumerate(data_pointings):
                if pointing[0][:8] == pointing2[:8]:
                    t.append(p_i2)
                    break
        r_indices.append(t)
        
    r_indices = np.array(r_indices)
    
    return r_indices, dets

pointing_path = "./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/0374/pre_ppc"

with open(f"{pointing_path}/pointings.pickle", "rb") as f:
    ps_pointings = pickle.load(f)

index_table, _ = pointing_indices_table(data_path, ps_pointings)



with open(f"{data_path_d}/counts.pickle", "wb") as f:
    pickle.dump((total_counts, energy_bins, dets, pointings), f)