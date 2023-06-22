import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

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
import pickle
# from MultinestClusterFit import powerlaw_binned_spectrum
# from astromodels import Powerlaw,  PointSource, SpectralComponent



path_smf = "./main_files/spimodfit_comparison_sim_source/smf_real_bkg/0374"
with fits.open(f"{path_smf}/energy_boundaries.fits.gz") as file:
    t = Table.read(file[1])
final_e_bins = np.append(t["E_MIN"], t["E_MAX"][-1])



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

def rebin_counts(counts, e_bins):
    e_indices = []
    for e in final_e_bins:
        temp = np.argwhere(e_bins==e)
        if len(temp)>0:
            e_indices.append(temp[0][0])    
    counts_binned = np.zeros((len(counts), len(final_e_bins)-1))
    for i in range(len(e_indices)-1):
        counts_binned[:,i] = np.sum(counts[ : , e_indices[i] : e_indices[i+1]], axis=1)
        
    return counts_binned



def calc_var_ratio_size(index_table, counts, lifetimes, dets):
    ratios = []

    for combination in index_table:
        indices1 = [85*combination[0] + i for i in dets]
        indices2 = [85*combination[1] + i for i in dets]
        
        counts1 = counts[indices1]
        counts2 = counts[indices2]
        
        
        mean = (counts1 + counts2) / 2
        variance = (counts1 - mean)**2 + (counts2 - mean)**2
        
        times1 = lifetimes[indices1][:,np.newaxis]
        times2 = lifetimes[indices2][:,np.newaxis]
        
        rate = (counts1 + counts2) / (times1 + times2)
        exp_var = (rate*times1 - mean)**2 + (rate*times2 - mean)**2 + rate*(times1 + times2) / 2
        
        rat = variance / exp_var
        
        ratios += list(rat.flatten())
        
    return np.array(ratios)




def setup_var_ratio_calculation(data_path, pointing_path):
    with open(f"{pointing_path}/pointings.pickle", "rb") as f:
        pointings = pickle.load(f)
        
    with fits.open(f"{data_path}/energy_boundaries.fits") as file:
        t = Table.read(file[1])
        e_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
        
    with fits.open(f"{data_path}/dead_time.fits") as file:
        t = Table.read(file[1])
        lifetimes = np.array(t["LIVETIME"])
        
    with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
        t = Table.read(file[1])
        counts = rebin_counts(t["COUNTS"], e_bins)
        
    index_table, dets = pointing_indices_table(data_path, pointings)
    
    ratios = calc_var_ratio_size(index_table, counts, lifetimes, dets)
    
    return ratios



pointing_path_0374_norm_pre = "./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/0374/pre_ppc"
pointing_path_0374_norm_post = "./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/0374/post_ppc"

pointing_path_0374_far_pre = "./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/0374/pre_ppc_far"
pointing_path_0374_far_post = "./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/0374/post_ppc_far"

pointing_path_0374_triple_pre = "./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/0374/pre_ppc_triple"
pointing_path_0374_triple_post = "./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/0374/post_ppc_triple"

pointing_path_1662_pre = "./main_files/ppc_test/ppc_test_1662_correct"
pointing_path_1662_post = "./main_files/ppc_test/ppc_test_1662_correct_wo_outliers"

data_path_0374 = "./main_files/SPI_data/0374"
data_path_1662 = "./main_files/SPI_data/1662"

num_bins=40
x_locs = [i for i in range(num_bins)]



def log_hist(a, bins=None):
    if bins is None:
        x_max = a.max()**(1/3)
        bins = np.linspace(0, x_max, num_bins) ** 3
    else:
        bins = bins
    hist, _ = np.histogram(a, bins)
    hist = hist / np.sum(hist)
    return hist, bins

def interpolate_linear(x1, x2, y1, y2, x):
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)

def x_ticks(bins):
    tick_labels = np.array([0, 1/8, 1, 1.5**3, 8, 2.5**3])
    tick_labels = np.linspace(0, 6, 13)**3
    tick_labels = tick_labels[:np.searchsorted(tick_labels, bins[-1])]
    tick_locs = tick_labels **(1/3)
    
    bins = bins ** (1/3)
    
    plt_tick_locs = []
    for i in range(len(tick_locs)):
        plt_tick_locs.append(interpolate_linear(bins[0], bins[-1], x_locs[0], x_locs[-1], tick_locs[i]))
                
    return np.array(plt_tick_locs), tick_labels
    
    
    
        
fig, ax = plt.subplots(nrows=5, figsize=(7,9))
fig.tight_layout()



ratios = setup_var_ratio_calculation(data_path_0374, pointing_path_0374_norm_pre)
hist, bins = log_hist(ratios)
ax[0].stairs(hist, x_locs, lw=4.0, color="C0", label=f"Pre-PPC: Mean={np.average(ratios):.3f}")

ratios = setup_var_ratio_calculation(data_path_0374, pointing_path_0374_norm_post)
hist, bins = log_hist(ratios, bins)
ax[0].stairs(hist, x_locs, lw=2., color="C1", label=f"Post-PPC: Mean={np.average(ratios):.3f}")

plt_tick_locs, tick_labels = x_ticks(bins)
ax[0].set_xticks(plt_tick_locs, tick_labels)
ax[0].set_title("0374 Real Data, $1.5^\circ$ Minimum Separation", fontsize=10, pad=1)



ratios = setup_var_ratio_calculation(data_path_0374, pointing_path_0374_far_pre)
hist, bins = log_hist(ratios)
ax[1].stairs(hist, x_locs, lw=4.0, color="C0", label=f"Pre-PPC: Mean={np.average(ratios):.3f}")

ratios = setup_var_ratio_calculation(data_path_0374, pointing_path_0374_far_post)
hist, bins = log_hist(ratios, bins)
ax[1].stairs(hist, x_locs, lw=2., color="C1", label=f"Post-PPC: Mean={np.average(ratios):.3f}")

plt_tick_locs, tick_labels = x_ticks(bins)
ax[1].set_xticks(plt_tick_locs, tick_labels)
ax[1].set_title("0374 Real Data, $2.5^\circ$ Minimum Separation", fontsize=10, pad=1)





ratios = setup_var_ratio_calculation(data_path_1662, pointing_path_1662_pre)
hist, bins = log_hist(ratios)
ax[4].stairs(hist, x_locs, lw=4.0, color="C0", label=f"Pre-PPC: Mean={np.average(ratios):.3f}")

ratios = setup_var_ratio_calculation(data_path_1662, pointing_path_1662_post)
hist, bins = log_hist(ratios, bins)
ax[4].stairs(hist, x_locs, lw=2., color="C1", label=f"Post-PPC: Mean={np.average(ratios):.3f}")

plt_tick_locs, tick_labels = x_ticks(bins)
ax[4].set_xticks(plt_tick_locs, tick_labels)
ax[4].set_title("1662 Real Data", fontsize=10, pad=1)







fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("$\Sigma$var / $\Sigma$E(var)")
plt.ylabel("Proportion of Counts", labelpad=20)

ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[3].legend()
ax[4].legend()

path_d = "./main_files/background_analysis"
if not os.path.exists(f"{path_d}"):
    os.mkdir(f"{path_d}")
    
plt.savefig(f"{path_d}/background_variance.pdf", bbox_inches='tight')

