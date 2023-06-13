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
# from RebinningFunctions import *

def identical_repeats():
    # identical repeats

    data_path = "./main_files/SPI_data/0374"
    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2
    pointing_index = 1

    data_path_d = "./main_files/pure_simulation_tests/identical_repeats"

    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
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

            
    temp_path = f"{data_path_d}"        

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        
    os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
    os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

            
    hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
    hdu.writeto(f"{temp_path}/dead_time.fits")

    source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_piv, source_K, source_index)

    total_counts = background_counts.copy()
    total_counts["COUNTS"] += source_counts
            
    hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{temp_path}/evts_det_spec.fits")
    
def different_sources():
    data_path = "./main_files/SPI_data/0374"
    source_ra, source_dec = 10, -40
    source_piv = 100.
    source_Ks = [0.5e-4, 2e-4, 8e-4]
    source_indices = [-0.5, -2, -8]
    pointing_index = 1

    data_path_d = "./main_files/pure_simulation_tests/different_sources"

    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_Ks, source_indices), f)

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
    for i in range(len(source_Ks)):
        for j in range(len(source_indices)):
            
            temp_path = f"{data_path_d}/{i}_{j}"        
            
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
                
            os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
            os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

                    
            hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
            hdu.writeto(f"{temp_path}/dead_time.fits")
            
            source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_piv, source_Ks[i], source_indices[j])
            
            total_counts = background_counts.copy()
            total_counts["COUNTS"] += source_counts
                    
            hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
            hdu.writeto(f"{temp_path}/evts_det_spec.fits")
            
def second_source_i_1():
    # second source
    data_path = "./main_files/SPI_data/0374"
    # data_path = "crab_data/1380"
    # primary

    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2

    # secondary
    s_source_decs = [-40, -40.5, -41.5, -45, -65, -85]
    s_source_Ks = [0.1e-4, 0.3e-4, 1e-4]
    s_source_index = -1
    pointing_index = 1
    # pointing_index = 4


    data_path_d = "./main_files/pure_simulation_tests/second_source_i_1"
    # data_path_d = "simulated_data/1380_const_bkg_sec_source"


    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_K, source_index, s_source_decs, s_source_Ks, s_source_index), f)

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
    s_resp_mats = []

    for i in range(len(s_source_decs)):
        t_resp_mats = []
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
                sd = SPIDRM(rmfs[d], source_ra, s_source_decs[i])
                sds = np.append(sds, sd.matrix.T)
            t_resp_mats.append(sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1)))
        s_resp_mats.append(t_resp_mats)
    def calc_count_rates_sec(resp_mats, ra, dec, piv, K, index, s_resp_mats, s_dec, s_K, s_index):
        pl = Powerlaw()
        pl.piv = piv
        pl.K = K
        pl.index = index
        component1 = SpectralComponent("pl", shape=pl)
        source = PointSource("Test", ra=ra, dec=dec, components=[component1])
        
        spec = source(emod)
        spec_binned = powerlaw_binned_spectrum(emod, spec)
        
        s_pl = Powerlaw()
        s_pl.piv = piv
        s_pl.K = s_K
        s_pl.index = s_index
        component1 = SpectralComponent("pl", shape=s_pl)
        s_source = PointSource("Test", ra=ra, dec=s_dec, components=[component1])
        
        s_spec = s_source(emod)
        s_spec_binned = powerlaw_binned_spectrum(emod, s_spec)
        
        source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1))
        
        s_source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1))
            
        for p_i, pointing in enumerate(pointings):
            if skip_pointing[p_i]:
                continue
            
            resp_mat = resp_mats[p_i]
            
            count_rates = np.dot(spec_binned, resp_mat)
                    
            for d_i, d in enumerate(dets):
                index = p_i * 85 + d
                source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])
                            
            s_resp_mat = s_resp_mats[p_i]
            
            s_count_rates = np.dot(s_spec_binned, s_resp_mat)
                        
            for d_i, d in enumerate(dets):
                index = p_i * 85 + d
                s_source_counts[index,:] = np.random.poisson(s_count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])
                
        total_source_counts = source_counts + s_source_counts
        
        return total_source_counts
    for i in range(len(s_source_decs)):
        for j in range(len(s_source_Ks)):
            
            temp_path = f"{data_path_d}/{i}_{j}"        
            
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
                
            os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
            os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

                    
            hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
            hdu.writeto(f"{temp_path}/dead_time.fits")
            
            source_counts = calc_count_rates_sec(resp_mats, source_ra, source_dec, source_piv, source_K, source_index, s_resp_mats[i], s_source_decs[i], s_source_Ks[j], s_source_index)
            
            total_counts = background_counts.copy()
            total_counts["COUNTS"] = source_counts + total_counts["COUNTS"]
                    
            hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
            hdu.writeto(f"{temp_path}/evts_det_spec.fits")
            
def second_source_i_2():
    # second source
    data_path = "./main_files/SPI_data/0374"
    # data_path = "crab_data/1380"
    # primary

    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2

    # secondary
    s_source_decs = [-40, -40.5, -41.5, -45, -65, -85]
    s_source_Ks = [0.1e-4, 0.3e-4, 1e-4]
    s_source_index = -2
    pointing_index = 1
    # pointing_index = 4


    data_path_d = "./main_files/pure_simulation_tests/second_source_i_2"
    # data_path_d = "simulated_data/1380_const_bkg_sec_source"


    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_K, source_index, s_source_decs, s_source_Ks, s_source_index), f)

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
    s_resp_mats = []

    for i in range(len(s_source_decs)):
        t_resp_mats = []
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
                sd = SPIDRM(rmfs[d], source_ra, s_source_decs[i])
                sds = np.append(sds, sd.matrix.T)
            t_resp_mats.append(sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1)))
        s_resp_mats.append(t_resp_mats)
    def calc_count_rates_sec(resp_mats, ra, dec, piv, K, index, s_resp_mats, s_dec, s_K, s_index):
        pl = Powerlaw()
        pl.piv = piv
        pl.K = K
        pl.index = index
        component1 = SpectralComponent("pl", shape=pl)
        source = PointSource("Test", ra=ra, dec=dec, components=[component1])
        
        spec = source(emod)
        spec_binned = powerlaw_binned_spectrum(emod, spec)
        
        s_pl = Powerlaw()
        s_pl.piv = piv
        s_pl.K = s_K
        s_pl.index = s_index
        component1 = SpectralComponent("pl", shape=s_pl)
        s_source = PointSource("Test", ra=ra, dec=s_dec, components=[component1])
        
        s_spec = s_source(emod)
        s_spec_binned = powerlaw_binned_spectrum(emod, s_spec)
        
        source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1))
        
        s_source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1))
            
        for p_i, pointing in enumerate(pointings):
            if skip_pointing[p_i]:
                continue
            
            resp_mat = resp_mats[p_i]
            
            count_rates = np.dot(spec_binned, resp_mat)
                    
            for d_i, d in enumerate(dets):
                index = p_i * 85 + d
                source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])
                            
            s_resp_mat = s_resp_mats[p_i]
            
            s_count_rates = np.dot(s_spec_binned, s_resp_mat)
                        
            for d_i, d in enumerate(dets):
                index = p_i * 85 + d
                s_source_counts[index,:] = np.random.poisson(s_count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])
                
        total_source_counts = source_counts + s_source_counts
        
        return total_source_counts
    for i in range(len(s_source_decs)):
        for j in range(len(s_source_Ks)):
            
            temp_path = f"{data_path_d}/{i}_{j}"        
            
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
                
            os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
            os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

                    
            hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
            hdu.writeto(f"{temp_path}/dead_time.fits")
            
            source_counts = calc_count_rates_sec(resp_mats, source_ra, source_dec, source_piv, source_K, source_index, s_resp_mats[i], s_source_decs[i], s_source_Ks[j], s_source_index)
            
            total_counts = background_counts.copy()
            total_counts["COUNTS"] = source_counts + total_counts["COUNTS"]
                    
            hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
            hdu.writeto(f"{temp_path}/evts_det_spec.fits")
            
def second_source_i_3():
    # second source
    data_path = "./main_files/SPI_data/0374"
    # data_path = "crab_data/1380"
    # primary

    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2

    # secondary
    s_source_decs = [-40, -40.5, -41.5, -45, -65, -85]
    s_source_Ks = [0.1e-4, 0.3e-4, 1e-4]
    s_source_index = -3
    pointing_index = 1
    # pointing_index = 4


    data_path_d = "./main_files/pure_simulation_tests/second_source_i_3"
    # data_path_d = "simulated_data/1380_const_bkg_sec_source"


    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_K, source_index, s_source_decs, s_source_Ks, s_source_index), f)

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
    s_resp_mats = []

    for i in range(len(s_source_decs)):
        t_resp_mats = []
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
                sd = SPIDRM(rmfs[d], source_ra, s_source_decs[i])
                sds = np.append(sds, sd.matrix.T)
            t_resp_mats.append(sds.reshape((len(dets), len(emod)-1, len(energy_bins)-1)))
        s_resp_mats.append(t_resp_mats)
    def calc_count_rates_sec(resp_mats, ra, dec, piv, K, index, s_resp_mats, s_dec, s_K, s_index):
        pl = Powerlaw()
        pl.piv = piv
        pl.K = K
        pl.index = index
        component1 = SpectralComponent("pl", shape=pl)
        source = PointSource("Test", ra=ra, dec=dec, components=[component1])
        
        spec = source(emod)
        spec_binned = powerlaw_binned_spectrum(emod, spec)
        
        s_pl = Powerlaw()
        s_pl.piv = piv
        s_pl.K = s_K
        s_pl.index = s_index
        component1 = SpectralComponent("pl", shape=s_pl)
        s_source = PointSource("Test", ra=ra, dec=s_dec, components=[component1])
        
        s_spec = s_source(emod)
        s_spec_binned = powerlaw_binned_spectrum(emod, s_spec)
        
        source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1))
        
        s_source_counts = np.zeros((len(pointings)*85, len(energy_bins)-1))
            
        for p_i, pointing in enumerate(pointings):
            if skip_pointing[p_i]:
                continue
            
            resp_mat = resp_mats[p_i]
            
            count_rates = np.dot(spec_binned, resp_mat)
                    
            for d_i, d in enumerate(dets):
                index = p_i * 85 + d
                source_counts[index,:] = np.random.poisson(count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])
                            
            s_resp_mat = s_resp_mats[p_i]
            
            s_count_rates = np.dot(s_spec_binned, s_resp_mat)
                        
            for d_i, d in enumerate(dets):
                index = p_i * 85 + d
                s_source_counts[index,:] = np.random.poisson(s_count_rates[d_i,:] * time_elapsed[pointing_index*85 + d])
                
        total_source_counts = source_counts + s_source_counts
        
        return total_source_counts
    for i in range(len(s_source_decs)):
        for j in range(len(s_source_Ks)):
            
            temp_path = f"{data_path_d}/{i}_{j}"        
            
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
                
            os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
            os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

                    
            hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
            hdu.writeto(f"{temp_path}/dead_time.fits")
            
            source_counts = calc_count_rates_sec(resp_mats, source_ra, source_dec, source_piv, source_K, source_index, s_resp_mats[i], s_source_decs[i], s_source_Ks[j], s_source_index)
            
            total_counts = background_counts.copy()
            total_counts["COUNTS"] = source_counts + total_counts["COUNTS"]
                    
            hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
            hdu.writeto(f"{temp_path}/evts_det_spec.fits")
            
def energy_range():
    # energy ranges

    data_path = "./main_files/SPI_data/0374"
    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2

    energy_ranges = [(None, 80),
                    (None, 200),
                    (None, 1000),
                    (None, None),
                    (80, None),
                    (200, None),
                    (1000, None),
                    (30, 80)]
    pointing_index = 1

    data_path_d = "./main_files/pure_simulation_tests/energy_range"

    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_K, source_index, energy_ranges), f)

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

            
    temp_path = f"{data_path_d}"        

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        
    os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
    os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

            
    hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
    hdu.writeto(f"{temp_path}/dead_time.fits")

    source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_piv, source_K, source_index)

    total_counts = background_counts.copy()
    total_counts["COUNTS"] += source_counts
            
    hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{temp_path}/evts_det_spec.fits")

def num_e_bins():
    # number of energy bins

    data_path = "./main_files/SPI_data/0374"
    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2

    bin_number = [5, 25, 125, 600]
    pointing_index = 1

    data_path_d = "./main_files/pure_simulation_tests/num_e_bins"

    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_K, source_index, bin_number), f)

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

            
    temp_path = f"{data_path_d}"        

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        
    os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
    os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

            
    hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
    hdu.writeto(f"{temp_path}/dead_time.fits")

    source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_piv, source_K, source_index)

    total_counts = background_counts.copy()
    total_counts["COUNTS"] += source_counts
            
    hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{temp_path}/evts_det_spec.fits")

def cluster_size():
    # cluster sizes

    data_path = "./main_files/SPI_data/0374"
    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2

    cluster_sizes = [2, 3, 4]
    pointing_index = 1

    data_path_d = "./main_files/pure_simulation_tests/cluster_size"

    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_K, source_index, cluster_sizes), f)

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

            
    temp_path = f"{data_path_d}"        

    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
        
    os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
    os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

            
    hdu = fits.BinTableHDU(data=updated_time, name="SPI.-OBS.-DTI") # is all of this correct?
    hdu.writeto(f"{temp_path}/dead_time.fits")

    source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_piv, source_K, source_index)

    total_counts = background_counts.copy()
    total_counts["COUNTS"] += source_counts
            
    hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
    hdu.writeto(f"{temp_path}/evts_det_spec.fits")

def data_scaling():
    # length of SCWs

    data_path = "./main_files/SPI_data/0374"
    source_ra, source_dec = 10, -40
    source_piv = 200.
    source_K = 1e-4
    source_index = -2

    time_fraction = [1, 0.3, 0.1]
    pointing_index = 1

    data_path_d = "./main_files/pure_simulation_tests/data_scaling"

    if not os.path.exists(f"{data_path_d}"):
        os.mkdir(f"{data_path_d}")
        
    with open(f"{data_path_d}/source_params.pickle", "wb") as f:
        pickle.dump((source_ra, source_dec, source_piv, source_K, source_index, time_fraction), f)

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

    b_cs = []

    for frac in range(len(time_fraction)):

        background_counts = t.copy()

        for i in range(int(len(background_counts) / 85)):
            if i == pointing_index:
                continue
            else:
                background_counts[i*85 : (i+1)*85] = background_counts[pointing_index*85 : (pointing_index+1)*85]
                
        background_counts["COUNTS"] = np.random.poisson(background_counts["COUNTS"] * time_fraction[frac])
        b_cs.append(background_counts)
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
    def calc_count_rates(resp_mats, ra, dec, piv, K, index, time_elapsed):
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
    for i in range(len(time_fraction)):
            
        temp_path = f"{data_path_d}/t_{i}"        

        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
            
        os.popen(f"cp {data_path}/energy_boundaries.fits {temp_path}/energy_boundaries.fits")
        os.popen(f"cp {data_path}/pointing.fits {temp_path}/pointing.fits")

        temp_time = updated_time.copy()
        temp_time["LIVETIME"] *= time_fraction[i]
        hdu = fits.BinTableHDU(data=temp_time, name="SPI.-OBS.-DTI") # is all of this correct?
        hdu.writeto(f"{temp_path}/dead_time.fits")

        source_counts = calc_count_rates(resp_mats, source_ra, source_dec, source_piv, source_K, source_index, time_elapsed*time_fraction[i])

        total_counts = b_cs[i].copy()
        total_counts["COUNTS"] += source_counts
                
        hdu = fits.BinTableHDU(data=total_counts, name="SPI.-OBS.-DSP")
        hdu.writeto(f"{temp_path}/evts_det_spec.fits")

# identical_repeats()
# different_sources()
# second_source_i_1()
# second_source_i_2()
# second_source_i_3()
# energy_range()
# num_e_bins()
# cluster_size()
# data_scaling()
