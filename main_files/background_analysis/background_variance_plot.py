import numpy as np
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from astropy.table import Table
from datetime import datetime
import astropy.time as at
from pyspi.utils.function_utils import find_response_version
# from pyspi.utils.response.spi_response_data import ResponseDataRMF
# from pyspi.utils.response.spi_response import ResponseRMFGenerator
# from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
import os
import pickle
# from MultinestClusterFit import powerlaw_binned_spectrum
# from astromodels import Powerlaw,  PointSource, SpectralComponent

np.random.seed(0)

path_c = "sim_source_0374_w_smf_bkg"
path_r = "crab_data/0374"
path_smf = "crab_data/0374_spimodfit_bkg"
path_smf_p = "crab_data/0374_spimodfit"

with open(f"./{path_c}/pointings.pickle", "rb") as f:
    d = pickle.load(f)
    

with fits.open(f"{path_r}/pointing.fits") as file:
    t = Table.read(file[1])
    
    pointings = np.array(t["PTID_SPI"])
    
    time_start = np.array(t["TSTART"]) + 2451544.5
    time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
    time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])

assert find_response_version(time_start[0]) == find_response_version(time_start[-1]), "Versions not constant"
version = find_response_version(time_start[0])
dets = get_live_dets(time=time_start[0], event_types=["single"])

r_indices = []
for c_i, cluster in enumerate(d):
    t = []
    for p_i, pointing in enumerate(cluster):
        for p_i2, pointing2 in enumerate(pointings):
            if pointing[0][:8] == pointing2[:8]:
                t.append(p_i2)
                break
    r_indices.append(t)
    
r_indices = np.array(r_indices)
    
with fits.open(f"{path_smf_p}/pointing.fits.gz") as file:
    t = Table.read(file[1])
    
    pointings_smf = np.array(t["PTID_SPI"])

smf_indices = []
for c_i, cluster in enumerate(d):
    t = []
    for p_i, pointing in enumerate(cluster):
        for p_i2, pointing2 in enumerate(pointings_smf):
            if pointing[0][:8] == pointing2[:8]:
                t.append(p_i2)
                break
    smf_indices.append(t)
    
smf_indices = np.array(smf_indices)

with fits.open(f"{path_r}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    r_e_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
with fits.open(f"{path_smf}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    smf_e_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
with fits.open(f"{path_r}/dead_time.fits") as file:
    t = Table.read(file[1])
    r_time_elapsed = np.array(t["LIVETIME"])

with fits.open(f"{path_smf}/dead_time.fits") as file:
    t = Table.read(file[1])
    smf_time_elapsed = np.array(t["LIVETIME"])

e_indices = []
for e in smf_e_bins:
    temp = np.argwhere(r_e_bins==e)
    if len(temp)>0:
        e_indices.append(temp[0][0])


with fits.open(f"{path_r}/evts_det_spec_orig.fits") as file:
    t = Table.read(file[1])
    r_counts = t["COUNTS"]
    
r_counts_binned = np.zeros((len(r_counts), len(smf_e_bins)-1))
for i in range(len(e_indices)-1):
    r_counts_binned[:,i] = np.sum(r_counts[ : , e_indices[i] : e_indices[i+1]], axis=1)

r_var = []
r_mean = []
r_e_var = []

for combination in r_indices:
    indices1 = [85*combination[0] + i for i in dets]
    indices2 = [85*combination[1] + i for i in dets]
    
    counts1 = r_counts_binned[indices1]
    counts2 = r_counts_binned[indices2]
    
    
    mean = (counts1 + counts2) / 2
    variance = (counts1 - mean)**2 + (counts2 - mean)**2
    
    times1 = r_time_elapsed[indices1][:,np.newaxis]
    times2 = r_time_elapsed[indices2][:,np.newaxis]
    
    rate = (counts1 + counts2) / (times1 + times2)
    e_var = (rate*times1 - mean)**2 + (rate*times2 - mean)**2 + rate*(times1 + times2) / 2
    
    
    r_mean.append(np.sum(mean))
    r_var.append(np.sum(variance))
    r_e_var.append(np.sum(e_var))
    
r_var = np.array(r_var)
r_mean = np.array(r_mean)
r_ratio = r_var/r_mean
r_e_var = np.array(r_e_var)
r_var_ratio = r_var / r_e_var


with fits.open(f"{path_smf}/bg-e0020-0600/output_bgmodel-conti.fits.gz") as file:
    t = Table.read(file[1])
    conti = t["COUNTS"]
with fits.open(f"{path_smf}/bg-e0020-0600/output_bgmodel-lines.fits.gz") as file:
    t = Table.read(file[1])
    lines = t["COUNTS"]


smf_var = []
smf_mean = []
smf_e_var = []


for combination in smf_indices:
    indices1 = [19*combination[0] + i for i in dets]
    indices2 = [19*combination[1] + i for i in dets]
    
    counts1 = (np.random.poisson(np.abs(lines[indices1])) * np.sign(lines[indices1])
                + np.random.poisson(np.abs(conti[indices1])) * np.sign(conti[indices1]))
    counts2 = (np.random.poisson(np.abs(lines[indices2])) * np.sign(lines[indices2])
                + np.random.poisson(np.abs(conti[indices2])) * np.sign(conti[indices2]))
    
    mean = (counts1 + counts2) / 2
    variance = (counts1 - mean)**2 + (counts2 - mean)**2
    size = np.size(counts1)
    
    indices1 = [85*combination[0] + i for i in dets]
    indices2 = [85*combination[1] + i for i in dets]
    
    times1 = smf_time_elapsed[indices1][:,np.newaxis]
    times2 = smf_time_elapsed[indices2][:,np.newaxis]
    
    rate = (counts1 + counts2) / (times1 + times2)
    e_var = (rate*times1 - mean)**2 + (rate*times2 - mean)**2 + rate*(times1 + times2) / 2
    
    smf_mean.append(np.sum(mean))
    smf_var.append(np.sum(variance))
    smf_e_var.append(np.sum(e_var))
    
smf_var = np.array(smf_var)
smf_mean = np.array(smf_mean)
smf_ratio = smf_var/smf_mean
smf_e_var = np.array(smf_e_var)
smf_var_ratio = smf_var / smf_e_var






path_crab = "crab_data/1662"
path_crab_p = "orbit_1662"

with open(f"./{path_crab_p}/pointings.pickle", "rb") as f:
    d = pickle.load(f)

with fits.open(f"{path_crab}/pointing.fits") as file:
    t = Table.read(file[1])
    
    pointings_crab = np.array(t["PTID_SPI"])
    
    time_start_crab = np.array(t["TSTART"]) + 2451544.5
    time_start_crab = [at.Time(f"{i}", format="jd").datetime for i in time_start_crab]
    time_start_crab = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start_crab])

assert find_response_version(time_start_crab[0]) == find_response_version(time_start_crab[-1]), "Versions not constant"
version = find_response_version(time_start_crab[0])
dets_crab = get_live_dets(time=time_start_crab[0], event_types=["single"])

crab_indices = []
for c_i, cluster in enumerate(d):
    t = []
    for p_i, pointing in enumerate(cluster):
        for p_i2, pointing2 in enumerate(pointings_crab):
            if pointing[0][:8] == pointing2[:8]:
                t.append(p_i2)
                break
    crab_indices.append(t)


with fits.open(f"{path_crab}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    c_e_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
with fits.open(f"{path_crab}/dead_time.fits") as file:
    t = Table.read(file[1])
    c_time_elapsed = np.array(t["LIVETIME"])

c_e_indices = []
for e in smf_e_bins:
    temp = np.argwhere(c_e_bins==e)
    if len(temp)>0:
        c_e_indices.append(temp[0][0])


with fits.open(f"{path_crab}/evts_det_spec.fits") as file:
    t = Table.read(file[1])
    c_counts = t["COUNTS"]
    
c_counts_binned = np.zeros((len(c_counts), len(smf_e_bins)-1))
for i in range(len(c_e_indices)-1):
    c_counts_binned[:,i] = np.sum(c_counts[ : , c_e_indices[i] : c_e_indices[i+1]], axis=1)

c_var = []
c_mean = []
c_e_var = []

for combination in crab_indices:
    indices1 = [85*combination[0] + i for i in dets_crab]
    indices2 = [85*combination[1] + i for i in dets_crab]
    
    counts1 = c_counts_binned[indices1]
    counts2 = c_counts_binned[indices2]
    
    
    mean = (counts1 + counts2) / 2
    variance = (counts1 - mean)**2 + (counts2 - mean)**2
    
    times1 = c_time_elapsed[indices1][:,np.newaxis]
    times2 = c_time_elapsed[indices2][:,np.newaxis]
    
    rate = (counts1 + counts2) / (times1 + times2)
    e_var = (rate*times1 - mean)**2 + (rate*times2 - mean)**2 + rate*(times1 + times2) / 2
    
    
    c_mean.append(np.sum(mean))
    c_var.append(np.sum(variance))
    c_e_var.append(np.sum(e_var))
    
c_var = np.array(c_var)
c_mean = np.array(c_mean)
c_ratio = c_var/c_mean
c_e_var = np.array(c_e_var)
c_var_ratio = c_var / c_e_var



path_crab2 = "crab_data/1662"

with open(f"./{path_crab_p}/pointings.pickle", "rb") as f:
    pointings2 = pickle.load(f)

d = []

bad_pointings = (
    "166200030010",
    "166200040010",
    "166200270010",
    "166200290010",
    "166200450010",
    "166200460010",
    "166200470010",
    "166200520010",
    "166200480010",
    "166200510010",
    "166200490010",
    "166200500010",
    "166200490020",
    "166200540010",
    "166200550010",
    "166200560010",
)
for cluster in pointings2:
    if cluster[0][0] in bad_pointings:
        continue
    else:
        d.append(cluster)
        
d = tuple(d)



with fits.open(f"{path_crab2}/pointing.fits") as file:
    t = Table.read(file[1])
    
    pointings_crab2 = np.array(t["PTID_SPI"])
    
    time_start_crab2 = np.array(t["TSTART"]) + 2451544.5
    time_start_crab2 = [at.Time(f"{i}", format="jd").datetime for i in time_start_crab2]
    time_start_crab2 = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start_crab2])

assert find_response_version(time_start_crab2[0]) == find_response_version(time_start_crab2[-1]), "Versions not constant"
version = find_response_version(time_start_crab2[0])
dets_crab2 = get_live_dets(time=time_start_crab2[0], event_types=["single"])

crab_indices2 = []
for c_i, cluster in enumerate(d):
    t = []
    for p_i, pointing in enumerate(cluster):
        for p_i2, pointing2 in enumerate(pointings_crab2):
            if pointing[0][:8] == pointing2[:8]:
                t.append(p_i2)
                break
    crab_indices2.append(t)


with fits.open(f"{path_crab2}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    c_e_bins2 = np.append(t["E_MIN"], t["E_MAX"][-1])
    
with fits.open(f"{path_crab2}/dead_time.fits") as file:
    t = Table.read(file[1])
    c_time_elapsed2 = np.array(t["LIVETIME"])

c_e_indices2 = []
for e in smf_e_bins:
    temp = np.argwhere(c_e_bins2==e)
    if len(temp)>0:
        c_e_indices2.append(temp[0][0])


with fits.open(f"{path_crab2}/evts_det_spec.fits") as file:
    t = Table.read(file[1])
    c_counts2 = t["COUNTS"]
    
c_counts_binned2 = np.zeros((len(c_counts2), len(smf_e_bins)-1))
for i in range(len(c_e_indices)-1):
    c_counts_binned2[:,i] = np.sum(c_counts2[ : , c_e_indices2[i] : c_e_indices2[i+1]], axis=1)

c_var2 = []
c_mean2 = []
c_e_var2 = []

for combination in crab_indices2:
    indices1 = [85*combination[0] + i for i in dets_crab2]
    indices2 = [85*combination[1] + i for i in dets_crab2]
    
    counts1 = c_counts_binned[indices1]
    counts2 = c_counts_binned[indices2]
    
    
    mean = (counts1 + counts2) / 2
    variance = (counts1 - mean)**2 + (counts2 - mean)**2
    
    times1 = c_time_elapsed[indices1][:,np.newaxis]
    times2 = c_time_elapsed[indices2][:,np.newaxis]
    
    rate = (counts1 + counts2) / (times1 + times2)
    e_var = (rate*times1 - mean)**2 + (rate*times2 - mean)**2 + rate*(times1 + times2) / 2
    
    
    c_mean2.append(np.sum(mean))
    c_var2.append(np.sum(variance))
    c_e_var2.append(np.sum(e_var))
    
c_var2 = np.array(c_var2)
c_mean2 = np.array(c_mean2)
c_ratio2 = c_var2/c_mean2
c_e_var2 = np.array(c_e_var2)
c_var_ratio2 = c_var2 / c_e_var2




# print(r_ratio)
print(r_var_ratio)
# print(smf_ratio)
print(smf_var_ratio)
print(c_var_ratio)
print(c_var_ratio2)


fig, ax = plt.subplots(nrows=4, figsize=(7,7))


ax[0].hist(r_var_ratio, bins=30, label="Real Background 0374")
ax[0].axvline(np.average(r_var_ratio), label="Mean", color="C1")

ax[1].hist(smf_var_ratio, bins=30, label="Spimodfit Background 0374")
ax[1].axvline(np.average(smf_var_ratio), label="Mean", color="C1")

ax[2].hist(c_var_ratio, bins=30, label="Real Crab 1662")
ax[2].axvline(np.average(c_var_ratio), label="Mean", color="C1")

ax[3].hist(c_var_ratio2, bins=30, label="Real Crab 1662 after PPC")
ax[3].axvline(np.average(c_var_ratio2), label="Mean", color="C1")

fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("$\Sigma$var / $\Sigma$E(var)")
plt.ylabel("Frequency")

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()

path_d = "simulated_data/background_analysis"
if not os.path.exists(f"{path_d}"):
    os.mkdir(f"{path_d}")
    
plt.savefig(f"{path_d}/background_variance.pdf")