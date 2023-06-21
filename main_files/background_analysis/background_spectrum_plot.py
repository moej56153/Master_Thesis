import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.table import Table
from datetime import datetime
import astropy.time as at
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
from MultinestClusterFit import powerlaw_binned_spectrum
from astromodels import Powerlaw,  PointSource, SpectralComponent

data_path = "./main_files/SPI_data/0374"
data_path_smf = "./main_files/spimodfit_comparison_sim_source/smf_real_bkg/0374"
data_path_smf_bkg = "./main_files/spimodfit_comparison_sim_source/pyspi_smf_bkg/0374/bg-e0020-0600"


# real data
pointing_index = 1

with fits.open(f"{data_path}/pointing.fits") as file:
    t = Table.read(file[1])
    
    pointings = np.array(t["PTID_SPI"])
    
    time_start = np.array(t["TSTART"]) + 2451544.5
    time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
    time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
    
print(pointings[pointing_index])
    
with fits.open(f"{data_path}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
    t = Table.read(file[1])
    counts = t["COUNTS"]
    
with fits.open(f"{data_path}/dead_time.fits") as file:
    t = Table.read(file[1])
    time_elapsed = np.array(t["LIVETIME"])

time = time_start[pointing_index]
version = find_response_version(time)
rsp_base = ResponseDataRMF.from_version(version)
dets = get_live_dets(time=time, event_types=["single"])

print(dets)

indices = [pointing_index*85 + i for i in dets]
background_counts = counts[indices]
background_counts_max = np.amax(background_counts, axis=0)
background_counts_min = np.amin(background_counts, axis=0)
background_counts_mean = np.average(background_counts, axis=0)



ra, dec = 10, -40
K, piv, index = 0.0045, 40, -2
    
pl = Powerlaw()
pl.piv = piv
pl.K = K
pl.index = index
component1 = SpectralComponent("pl", shape=pl)
source = PointSource("Test", ra=ra, dec=dec, components=[component1])

emod = np.geomspace(10, 3000, 50)
spec = source(emod)
spec_binned = powerlaw_binned_spectrum(emod, spec)

rmfs = []
for d in dets:
    rmfs.append(ResponseRMFGenerator.from_time(time, d, energy_bins, emod, rsp_base))

sds = np.empty(0)
for d in range(len(dets)):
    sd = SPIDRM(rmfs[d], ra, dec)
    sds = np.append(sds, sd.matrix.T)
resp_mat = sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1))

count_rates = np.dot(spec_binned, resp_mat)

source_counts = np.zeros((len(dets), len(energy_bins)-1))
for d_i, d in enumerate(dets):
    index = pointing_index * 85 + d
    source_counts[d_i,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[index])

source_counts_max = np.amax(source_counts, axis=0)
source_counts_min = np.amin(source_counts, axis=0)
source_counts_mean = np.average(source_counts, axis=0)



# smf data
with fits.open(f"{data_path_smf}/pointing.fits.gz") as file:
    t = Table.read(file[1])
    
    pointings_smf = np.array(t["PTID_SPI"])
    
    time_start_smf = np.array(t["TSTART"]) + 2451544.5
    time_start_smf = [at.Time(f"{i}", format="jd").datetime for i in time_start_smf]
    time_start_smf = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start_smf])
    
print(pointings_smf[pointing_index])
    
with fits.open(f"{data_path_smf}/energy_boundaries.fits.gz") as file:
    t = Table.read(file[1])
    energy_bins_smf = np.append(t["E_MIN"], t["E_MAX"][-1])
    
    
with fits.open(f"{data_path_smf}/dead_time.fits.gz") as file:
    t = Table.read(file[1])
    time_elapsed_smf = np.array(t["LIVETIME"])
    
e_indices = []
for e in energy_bins_smf:
    temp = np.argwhere(energy_bins==e)
    if len(temp)>0:
        e_indices.append(temp[0][0])
        
# print(energy_bins_smf)

binned_background_counts = np.empty((len(dets), len(e_indices)-1))
for i in range(len(e_indices)-1):
    binned_background_counts[:,i] = np.sum(background_counts[ : , e_indices[i] : e_indices[i+1]], axis=1)
    
binned_background_counts_max = np.amax(binned_background_counts, axis=0)
binned_background_counts_min = np.amin(binned_background_counts, axis=0)
binned_background_counts_mean = np.average(binned_background_counts, axis=0)

with fits.open(f"{data_path_smf_bkg}/output_bgmodel-conti.fits.gz") as file:
    t = Table.read(file[1])
    conti = t["COUNTS"]
with fits.open(f"{data_path_smf_bkg}/output_bgmodel-lines.fits.gz") as file:
    t = Table.read(file[1])
    lines = t["COUNTS"]



smf_indices = [19*pointing_index + i for i in dets]

# print(lines[smf_indices])
# print(conti[smf_indices])
# print(binned_background_counts)
# print(time_elapsed[smf_indices])

smf_background_counts = (np.random.poisson(np.abs(lines[smf_indices])) * np.sign(lines[smf_indices])
                        + np.random.poisson(np.abs(conti[smf_indices])) * np.sign(conti[smf_indices]))
smf_background_counts_max = np.amax(smf_background_counts, axis=0)
smf_background_counts_min = np.amin(smf_background_counts, axis=0)
smf_background_counts_mean = np.average(smf_background_counts, axis=0)

fig, ax = plt.subplots(nrows=2, figsize=(8,8))

ax[0].step(energy_bins[:-1], background_counts_mean, color="r", where="post", label="Real Background")
ax[0].fill_between(energy_bins[:-1], background_counts_min, background_counts_max, step="post", color="r", alpha=0.3)
# plt.step(energy_bins[:-1], background_counts_min, color="r", lw=0.1, alpha=0.3)
# plt.step(energy_bins[:-1], background_counts_max, color="r", lw=0.1, alpha=0.3)

ax[0].step(energy_bins[:-1], source_counts_mean, color="g", where="post", label="Simulated Crab-like Source")
ax[0].fill_between(energy_bins[:-1], source_counts_min, source_counts_max, step="post", color="g", alpha=0.3)

ax[0].set_yscale("symlog")
ax[1].set_yscale("log")

plt.xlabel("Energy [keV]")
ax[0].set_ylabel("Counts")
ax[1].set_ylabel("Counts")



ax[1].step(energy_bins_smf[:-1], binned_background_counts_mean, color="r", where="post", label="Real Background")
ax[1].step(energy_bins_smf[-2:], np.repeat(binned_background_counts_mean[-1], 2), color="r", where="post")
ax[1].fill_between(energy_bins_smf[:-1], binned_background_counts_min, binned_background_counts_max, step="post", color="r", alpha=0.3)
ax[1].fill_between(energy_bins_smf[-2:], 
                   np.repeat(binned_background_counts_min[-1], 2), 
                   np.repeat(binned_background_counts_max[-1], 2), 
                   step="post", color="r", alpha=0.3)

ax[1].step(energy_bins_smf[:-1], smf_background_counts_mean, color="b", where="post", label="Spimodfit Background")
ax[1].step(energy_bins_smf[-2:], np.repeat(smf_background_counts_mean[-1], 2), color="b", where="post")
ax[1].fill_between(energy_bins_smf[:-1], smf_background_counts_min, smf_background_counts_max, step="post", color="b", alpha=0.3)
ax[1].fill_between(energy_bins_smf[-2:], 
                   np.repeat(smf_background_counts_min[-1], 2), 
                   np.repeat(smf_background_counts_max[-1], 2), 
                   step="post", color="b", alpha=0.3)

ax[0].legend()
ax[1].legend()


path_d = "./main_files/background_analysis"
if not os.path.exists(f"{path_d}"):
    os.mkdir(f"{path_d}")
    
plt.savefig(f"{path_d}/background_spectrum.pdf")