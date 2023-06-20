import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import astropy.io.fits as fits
from astropy.table import Table
from astromodels import Powerlaw,  PointSource, SpectralComponent
import astropy.time as at
from datetime import datetime
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
import os
from MultinestClusterFit import powerlaw_binned_spectrum

# put original data files from afs in this folder
# rename evts_det_spec.fits to evts_det_spec_orig.fits
# remember skip pointing for 1380, twice
# remember source parameters



revolution = "0374"


data_path = f"./main_files/SPI_data/{revolution}"
# data_path = "./main_files/SPI_data/1380"

ra, dec = 10, -40
K, piv, index = 6e-3, 40, -2
# ra, dec = 155., 75.
# K, piv, index = 3e-3, 40, -1

# Define  Spectrum
pl = Powerlaw()
pl.piv = piv
pl.K = K
pl.index = index
component1 = SpectralComponent("pl", shape=pl)
source = PointSource("Test", ra=ra, dec=dec, components=[component1])

emod = np.geomspace(18, 3000, 100)
spec = source(emod)
spec_binned = powerlaw_binned_spectrum(emod, spec)
# spec_binned = (emod[1:]-emod[:-1])*(spec[:-1]+spec[1:])/2


def calc_source_counts():
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
    pointings
    # Only necessary for 1380
    skip_pointing = [False] * len(pointings)
    skip_pointing[0] = True
    # Time Elapsed
    # det=i, pointing_index=j : index = j*85 + i
    with fits.open(f"{data_path}/dead_time.fits") as file:
        t = Table.read(file[1])
        time_elapsed = np.array(t["LIVETIME"])


    # Define  Spectrum
    pl = Powerlaw()
    pl.piv = piv
    pl.K = K
    pl.index = index
    component1 = SpectralComponent("pl", shape=pl)
    source = PointSource("Test", ra=ra, dec=dec, components=[component1])

    emod = np.geomspace(18, 3000, 100)
    spec = source(emod)
    spec_binned = powerlaw_binned_spectrum(emod, spec)
    # spec_binned = (emod[1:]-emod[:-1])*(spec[:-1]+spec[1:])/2
    # Generate Source Counts

    assert find_response_version(time_start[0]) == find_response_version(time_start[-1]), "Versions not constant"
    version = find_response_version(time_start[0])
    rsp_base = ResponseDataRMF.from_version(version)

    source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1), dtype=np.uint32)

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
            sd = SPIDRM(rmfs[d], ra, dec)
            sds = np.append(sds, sd.matrix.T)
        resp_mat = sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1))
        
        count_rates = np.dot(spec_binned, resp_mat)
        
        for d_i, d in enumerate(dets):
            index = p_i * 85 + d
            source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[index]) 
            
    return source_counts
    


def pyspi_real_bkg():
    destination_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{revolution}"
    
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
    pointings
    # Only necessary for 1380
    skip_pointing = [False] * len(pointings)
    # skip_pointing[0] = True ######################################################################
    
    
    # Time Elapsed
    # det=i, pointing_index=j : index = j*85 + i
    with fits.open(f"{data_path}/dead_time.fits") as file:
        t = Table.read(file[1])
        time_elapsed = np.array(t["LIVETIME"])


    
    # Generate Source Counts

    assert find_response_version(time_start[0]) == find_response_version(time_start[-1]), "Versions not constant"
    version = find_response_version(time_start[0])
    rsp_base = ResponseDataRMF.from_version(version)

    source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1), dtype=np.uint32)

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
            sd = SPIDRM(rmfs[d], ra, dec)
            sds = np.append(sds, sd.matrix.T)
        resp_mat = sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1))
        
        count_rates = np.dot(spec_binned, resp_mat)
        
        for d_i, d in enumerate(dets):
            index = p_i * 85 + d
            source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[index])    

    # Save Data for PySpi

    with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
        t = Table.read(file[1])
        
        counts = t
        
    updated_counts = counts.copy()
    updated_counts["COUNTS"] += source_counts

    hdu = fits.BinTableHDU(data=updated_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{destination_path}/evts_det_spec.fits")
    
    os.popen(f"cp {data_path}/energy_boundaries.fits {destination_path}/energy_boundaries.fits")
    os.popen(f"cp {data_path}/pointing.fits {destination_path}/pointing.fits")
    os.popen(f"cp {data_path}/dead_time.fits {destination_path}/dead_time.fits")


# pyspi_real_bkg()


# Sim source data for spimodfit
# put spimodfit spi/ files in this folder
# rename evts_det_spec.fits.gz to evts_det_spec_old.fits.gz

spimodfit_folder = f"./main_files/spimodfit_comparison_sim_source/smf_real_bkg/{revolution}"
# spimodfit_folder = "./main_files/spimodfit_comparison_sim_source/smf_real_bkg/1380"

# replace evts_det_spec.fits.gz in spimodfit to run fit

def spimodfit_real_bkg():
    if not os.path.exists(spimodfit_folder):
        os.mkdir(spimodfit_folder)
    with fits.open(f"{spimodfit_folder}/pointing.fits.gz") as file:
        t = Table.read(file[1])
        
    spimodfit_pointings = np.array(t["PTID_ISOC"])

    with fits.open(f"{spimodfit_folder}/energy_boundaries.fits.gz") as file:
        t = Table.read(file[1])
        
    spimodfit_energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])

    # Pointing Indices
    with fits.open(f"{data_path}/pointing.fits") as file:
        t = Table.read(file[1])
        p_indices = []
        p_rel_indices = []
        for i, p in enumerate(spimodfit_pointings):
            temp = np.argwhere(t["PTID_ISOC"]==p)
            if len(temp)>0:
                p_indices.append(temp[0][0])
                p_rel_indices.append(i)

    # Energy Indices
    with fits.open(f"{data_path}/energy_boundaries.fits") as file:
        t = Table.read(file[1])
        e_indices = []
        for e in spimodfit_energy_bins:
            temp = np.argwhere(t["E_MIN"]==e)
            if len(temp)>0:
                e_indices.append(temp[0][0])
                
    # Single Event Indices
    se_indices = []
    for i in p_indices:
        for j in range(19):
            se_indices.append(i*85 + j)
            
    # Relative Pointing Deterctor Indices
    rd_indices = []
    for i in p_rel_indices:
        for j in range(19):
            rd_indices.append(i*19 + j)
    # evts_det_spec.fits.gz
    with fits.open(f"{data_path}/evts_det_spec.fits") as file:
        t = Table.read(file[1])
        
    with fits.open(f"{spimodfit_folder}/evts_det_spec_old.fits.gz") as file:
        d = Table.read(file[1])
        h = file[1].header
        
    eds_temp = t[se_indices]


    for i in range(len(e_indices)-1):
        d["COUNTS"][:,i] = np.sum(eds_temp["COUNTS"][ : , e_indices[i] : e_indices[i+1]], axis=1)
        
    hdu = fits.BinTableHDU(data=d, name="SPI.-OBS.-DSP")
    hdu.header = h
    hdu.writeto(f"{spimodfit_folder}/evts_det_spec.fits.gz")
    
    







# Spimodfit backgroud
# put bg_e0020-0600/ in this folder

data_path2 = f"./main_files/spimodfit_comparison_sim_source/pyspi_smf_bkg/{revolution}"
smf_data_path2 = f"./main_files/spimodfit_comparison_sim_source/smf_smf_bkg/{revolution}"

def smf_bkg():
    os.popen(f"cp {spimodfit_folder}/energy_boundaries.fits {data_path2}/energy_boundaries.fits")
    
    if not os.path.exists(data_path2):
        os.mkdir(data_path2)
    with fits.open(f"{data_path2}/bg-e0020-0600/output_bgmodel-conti.fits.gz") as file:
        t = Table.read(file[1])
        conti = t["COUNTS"]
        
    with fits.open(f"{data_path2}/bg-e0020-0600/output_bgmodel-lines.fits.gz") as file:
        t = Table.read(file[1])
        lines = t["COUNTS"]
        
    with fits.open(f"{spimodfit_folder}/pointing.fits.gz") as file:
        t = Table.read(file[1])
        
    spimodfit_pointings = np.array(t["PTID_ISOC"])

    with fits.open(f"{spimodfit_folder}/energy_boundaries.fits.gz") as file:
        t = Table.read(file[1])
        
    spimodfit_energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])

    new_energy_bins = spimodfit_energy_bins
    # with fits.open(f"{data_path2}/energy_boundaries.fits") as file:
    #     t = Table.read(file[1])
    #     new_energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
    with fits.open(f"{data_path}/energy_boundaries.fits") as file:
        t = Table.read(file[1])
        energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
        
    # Energy Indices
    energy_indices = []
    for e in new_energy_bins:
        temp = np.argwhere(energy_bins==e)
        if len(temp)>0:
            energy_indices.append(temp[0][0])

    # Pointing Indices
    with fits.open(f"{data_path}/pointing.fits") as file:
        t = Table.read(file[1])
        pointing_indices = []
        pointing_indices_full = []
        for i, p in enumerate(spimodfit_pointings):
            temp = np.argwhere(t["PTID_ISOC"]==p)
            if len(temp)>0:
                pointing_indices.append(temp[0][0])
                for j in range(85):
                    pointing_indices_full.append(85*temp[0][0] + j)
                    
    
    source_counts = calc_source_counts()

    new_source_counts = np.zeros((len(source_counts), len(new_energy_bins)-1))

    for i in range(len(energy_indices)-1):
        new_source_counts[:,i] = np.sum(source_counts[ : , energy_indices[i] : energy_indices[i+1]], axis=1)
        
    new_source_counts = new_source_counts[pointing_indices_full]

    total_counts = new_source_counts.copy()

    for i in range(len(pointing_indices)):
        total_counts[85*i : 85*i + 19, :] += np.random.poisson(lines[i*19: (i+1)*19] + conti[i*19: (i+1)*19])
    with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
        t = Table.read(file[1])
        
        counts = t
        
    updated_counts = counts[pointing_indices_full]
    updated_counts["COUNTS"] = total_counts

    hdu = fits.BinTableHDU(data=updated_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{data_path2}/evts_det_spec.fits")
    # dead_time.fits
    with fits.open(f"{data_path}/dead_time.fits") as file:
        t = Table.read(file[1])
        
    dt = t[pointing_indices_full]

    hdu = fits.BinTableHDU(data=dt, name="SPI.-OBS.-DTI")
    hdu.writeto(f"{data_path2}/dead_time.fits")
    # pointing.fits
    with fits.open(f"{data_path}/pointing.fits") as file:
        t = Table.read(file[1])
        
    ps = t[pointing_indices]

    hdu = fits.BinTableHDU(data=ps, name="SPI.-OBS.-PNT")
    hdu.writeto(f"{data_path2}/pointing.fits")

    # spimodfit with spimodfit bkg

    if not os.path.exists(f"{spimodfit_folder}/w_smf_bkg"):
        os.mkdir(f"{spimodfit_folder}/w_smf_bkg")

    with fits.open(f"{data_path2}/evts_det_spec.fits") as file:
        t = Table.read(file[1])
        
    with fits.open(f"{spimodfit_folder}/evts_det_spec_old.fits.gz") as file:
        d = Table.read(file[1])
        h = file[1].header

    smf_bkg_indices = []
    for i in range(len(pointing_indices)):
        for j in range(19):
            smf_bkg_indices.append(i*85 + j)

    d["COUNTS"] = np.array(t["COUNTS"][smf_bkg_indices], dtype=np.uint32)

    hdu = fits.BinTableHDU(data=d, name="SPI.-OBS.-DSP")
    hdu.header = h
    hdu.writeto(f"{smf_data_path2}/evts_det_spec.fits.gz")
    
    os.popen(f"cp {spimodfit_folder}/energy_boundaries.fits {smf_data_path2}/energy_boundaries.fits")
    os.popen(f"cp {spimodfit_folder}/pointing.fits {smf_data_path2}/pointing.fits")
    os.popen(f"cp {spimodfit_folder}/dead_time.fits {smf_data_path2}/dead_time.fits")
    
    




# PySpi with constant background

pointing_index = 1 # 0374
# pointing_index = 4 # 1380

data_path3 = f"./main_files/spimodfit_comparison_sim_source/pyspi_const_bkg/{revolution}"
# data_path3 = "crab_data/1380_const_bkg"

def pyspi_const_bkg():

    if not os.path.exists(f"{data_path3}"):
        os.mkdir(f"{data_path3}")
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
            
    hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
    hdu.writeto(f"{data_path3}/dead_time.fits")
    # Generate Source Counts

    assert find_response_version(time_start[0]) == find_response_version(time_start[-1]), "Versions not constant"
    version = find_response_version(time_start[0])
    rsp_base = ResponseDataRMF.from_version(version)

    source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1), dtype=np.uint32)
    
    # Only necessary for 1380
    skip_pointing = [False] * len(pointings)
    # skip_pointing[0] = True ######################################################################

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
            sd = SPIDRM(rmfs[d], ra, dec)
            sds = np.append(sds, sd.matrix.T)
        resp_mat = sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1))
        
        count_rates = np.dot(spec_binned, resp_mat)
        
        for d_i, d in enumerate(dets):
            index = p_i * 85 + d
            source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])    
    # Save Data for PySpi

    with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
        t = Table.read(file[1])
        
        counts = t
        
    updated_counts = counts.copy()

    for i in range(int(len(updated_counts) / 85)):
        if i == pointing_index:
            continue
        else:
            updated_counts[i*85 : (i+1)*85] = updated_counts[pointing_index*85 : (pointing_index+1)*85]
            
    updated_counts["COUNTS"] = np.random.poisson(updated_counts["COUNTS"])

    updated_counts["COUNTS"] += source_counts

    hdu = fits.BinTableHDU(data=updated_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{data_path3}/evts_det_spec.fits")
    
    os.popen(f"cp {data_path}/energy_boundaries.fits {data_path3}/energy_boundaries.fits")
    os.popen(f"cp {data_path}/pointing.fits {data_path3}/pointing.fits")




