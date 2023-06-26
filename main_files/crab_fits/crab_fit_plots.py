import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.table import Table
import astropy.time as at
from datetime import datetime as dt
import time
import pickle

plot_path = "./main_files/crab_fits"

pyspi_fit_path = "./main_files/crab_fits/only_weak_pulsar"
smf_fit_path = "./main_files/spimodfit_fits"

data_path = "./main_files/SPI_data"

combined_fits_weak_pulsar = {
    "0043_4_5": ["0043", "0044", "0045"],
    "0422": ["0422"],
    "0665_6": ["0665", "0666"],
    "1268_9_78": ["1268", "1278"],
    "1327_8": ["1327"],
    "1515_6_20_8": ["1516", "1520", "1528"],
    "1657_8_61_2_4_7": ["1657", "1658", "1661", "1662", "1664",],
    "1781_4_5_9": ["1781", "1784", "1785", "1789"],
    "1921_5_7_30": ["1921", "1925", "1927", "1930"],
    "1996_9_2000": ["1996", "1999", "2000"],
    "2058_62_3_6": ["2058", "2062", "2063", "2066"],
}

def toYearFraction(date):
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def crab_low_energy_powerlaw_combined_plot(fit_dict=combined_fits_weak_pulsar):
    pyspi_fit_files = "pl_w_p/source_parameters.pickle"
    smf_fit_files = "crab_low_energy_pl_fit.pickle"
    
    years = []
    pyspi_indices = []
    pyspi_indices_errors = []
    
    pyspi_Ks = []
    pyspi_Ks_errors = []
    
    smf_indices = []
    smf_indices_errors = []
    
    smf_Ks = []
    smf_Ks_errors = []
        
    for folder, revolutions in fit_dict.items():
        with fits.open(f"{data_path}/{revolutions[0]}/pointing.fits") as file:
            t = Table.read(file[1])
            time_start = np.array(t["TSTART"]) + 2451544.5
            time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
        years.append(toYearFraction(time_start[-1]))
        
        with open(f"{pyspi_fit_path}/{folder}/{pyspi_fit_files}", "rb") as f:
            pyspi_val, pyspi_cov = pickle.load(f)
        
        pyspi_Ks.append(pyspi_val[0])
        pyspi_Ks_errors.append(pyspi_cov[0,0]**0.5)
        
        pyspi_indices.append(pyspi_val[1])
        pyspi_indices_errors.append(pyspi_cov[1,1]**0.5)
        
        with open(f"{smf_fit_path}/{folder}/{smf_fit_files}", "rb") as f:
            smf_val, smf_cov = pickle.load(f)
                
        smf_Ks.append(smf_val[0])
        smf_Ks_errors.append(smf_cov[0,0]**0.5)
        
        smf_indices.append(smf_val[1])
        smf_indices_errors.append(smf_cov[1,1]**0.5)
        
    fig, axes = plt.subplots(nrows=2)
    
    axes[0].errorbar(years, pyspi_Ks, pyspi_Ks_errors, label="PySPI")
    axes[0].errorbar(years, smf_Ks, smf_Ks_errors, label="Spimodfit")
    
    axes[1].errorbar(years, pyspi_indices, pyspi_indices_errors, label="PySPI")
    axes[1].errorbar(years, smf_indices, smf_indices_errors, label="Spimodfit")
    
    axes[1].legend()
    
    axes[1].set_xlabel("Year")
    axes[0].set_ylabel("Crab K\n(Normalization at 100keV)\n[keV$^{-1}$s$^{-1}$cm$^{-2}$]")
    axes[0].ticklabel_format(style='sci', axis="y",scilimits=(1,4))
    axes[1].set_ylabel("Crab index")
    
    plt.savefig(f"{plot_path}/crab_low_energy_pl.pdf", bbox_inches='tight')
    
        
crab_low_energy_powerlaw_combined_plot()