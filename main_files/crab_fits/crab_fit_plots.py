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
# pyspi_fit_path = "./main_files/crab_fits/strong_pulsar"

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

combined_fits_med_pulsar_sm = {
    "0043_4_5": ["0043", "0044", "0045"],
    "0422": ["0422"],
    "0665_6": ["0665", "0666"],
    "0966_7_70": ["0966", "0967", "0970"],
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
        
        if folder == "0966_7_70":
            pyspi_fit_path2 = "./main_files/crab_fits/strong_pulsar"
            with open(f"{pyspi_fit_path2}/{folder}/{pyspi_fit_files}", "rb") as f:
                pyspi_val, pyspi_cov = pickle.load(f)
        else:
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
        
    print(years)
    print(pyspi_Ks)
    print(smf_Ks)
        
    fig, axes = plt.subplots(nrows=2)
    
    axes[0].errorbar(years, pyspi_Ks, pyspi_Ks_errors, label="PySPI", capsize=2.5, linestyle='dotted')
    axes[0].errorbar(years, smf_Ks, smf_Ks_errors, label="Spimodfit", capsize=2.5, linestyle='dotted')
    
    axes[1].errorbar(years, pyspi_indices, pyspi_indices_errors, label="PySPI", capsize=2.5, linestyle='dotted')
    axes[1].errorbar(years, smf_indices, smf_indices_errors, label="Spimodfit", capsize=2.5, linestyle='dotted')
    
    axes[1].legend()
    
    axes[1].set_xlabel("Year")
    axes[0].set_ylabel("Crab K\n(Normalization at 100keV)\n[keV$^{-1}$s$^{-1}$cm$^{-2}$]")
    axes[0].ticklabel_format(style='sci', axis="y",scilimits=(1,4))
    axes[1].set_ylabel("Crab index")
    
    plt.savefig(f"{plot_path}/crab_low_energy_pl.pdf", bbox_inches='tight')


def crab_broken_powerlaw_100_combined_plot(fit_dict=combined_fits_weak_pulsar):
    pyspi_fit_files = "br_pl_100_w_p/source_parameters.pickle"
    smf_fit_files = "crab_brk_pl_100_fit.pickle"
    
    years = []
    pyspi_alphas = []
    pyspi_alphas_errors = []
    
    pyspi_betas = []
    pyspi_betas_errors = []
    
    pyspi_Ks = []
    pyspi_Ks_errors = []
    
    smf_alphas = []
    smf_alphas_errors = []
    
    smf_betas = []
    smf_betas_errors = []
    
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
        
        pyspi_alphas.append(pyspi_val[1])
        pyspi_alphas_errors.append(pyspi_cov[1,1]**0.5)
        
        pyspi_betas.append(pyspi_val[2])
        pyspi_betas_errors.append(pyspi_cov[2,2]**0.5)
        
        with open(f"{smf_fit_path}/{folder}/{smf_fit_files}", "rb") as f:
            smf_val, smf_cov = pickle.load(f)
                
        smf_Ks.append(smf_val[0])
        smf_Ks_errors.append(smf_cov[0,0]**0.5)
        
        smf_alphas.append(smf_val[1])
        smf_alphas_errors.append(smf_cov[1,1]**0.5)
        
        smf_betas.append(smf_val[2])
        smf_betas_errors.append(smf_cov[2,2]**0.5)
        
    paper_years = years[:3]
    
    paper_Ks = [6.6e-4, 6.65e-4, 6.7e-4]
    paper_Ks_errors = [0.1e-4, 0.1e-4, 0.1e-4]
    
    paper_alphas = [-2.07, -2.07, -2.06]
    paper_alphas_errors = [0.01, 0.01, 0.01]
    
    paper_betas = [-2.24, -2.25, -2.25]
    paper_betas_errors = [0.02, 0.03, 0.04]
        
    fig, axes = plt.subplots(nrows=3, figsize=(7, 7))
    
    axes[0].errorbar(years, pyspi_Ks, pyspi_Ks_errors, label="PySPI", capsize=2.5, linestyle='dotted')
    axes[0].errorbar(years, smf_Ks, smf_Ks_errors, label="Spimodfit", capsize=2.5, linestyle='dotted')
    axes[0].errorbar(paper_years, paper_Ks, paper_Ks_errors, label="Jourdain and Roques (2009)", capsize=2.5, linestyle='dotted')
    
    axes[1].errorbar(years, pyspi_alphas, pyspi_alphas_errors, label="PySPI", capsize=2.5, linestyle='dotted')
    axes[1].errorbar(years, smf_alphas, smf_alphas_errors, label="Spimodfit", capsize=2.5, linestyle='dotted')
    axes[1].errorbar(paper_years, paper_alphas, paper_alphas_errors, label="Jourdain and Roques (2009)", capsize=2.5, linestyle='dotted')
    
    axes[2].errorbar(years, pyspi_betas, pyspi_betas_errors, label="PySPI", capsize=4, linestyle='dotted')
    axes[2].errorbar(years, smf_betas, smf_betas_errors, label="Spimodfit", capsize=4, linestyle='dotted')
    axes[2].errorbar(paper_years, paper_betas, paper_betas_errors, label="Jourdain and Roques (2009)", capsize=4, linestyle='dotted')
    
    
    axes[2].set_xlabel("Year")
    axes[0].set_ylabel("Crab K\n(Normalization at 100keV)\n[keV$^{-1}$s$^{-1}$cm$^{-2}$]")
    axes[0].ticklabel_format(style='sci', axis="y",scilimits=(1,4))
    axes[1].set_ylabel("Crab $\\alpha$")
    axes[2].set_ylabel("Crab $\\beta$")
    
    lgd = axes[0].legend(bbox_to_anchor=(1.0, 1.5), loc=1, borderaxespad=0.)
    
    plt.savefig(f"{plot_path}/crab_brk_pl_100.pdf", bbox_inches='tight', bbox_extra_artists=(lgd,))

def crab_c_band_combined_plot(fit_dict=combined_fits_weak_pulsar):
    pyspi_fit_files = "lower_band_w_p/source_parameters.pickle"
    smf_fit_files = "crab_c_band_fit.pickle"
    
    years = []
    pyspi_alphas = []
    pyspi_alphas_errors = []
    
    
    pyspi_Ks = []
    pyspi_Ks_errors = []
    
    smf_alphas = []
    smf_alphas_errors = []
    
    
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
        
        pyspi_alphas.append(pyspi_val[1])
        pyspi_alphas_errors.append(pyspi_cov[1,1]**0.5)
        
        
        with open(f"{smf_fit_path}/{folder}/{smf_fit_files}", "rb") as f:
            smf_val, smf_cov = pickle.load(f)
                
        smf_Ks.append(smf_val[0])
        smf_Ks_errors.append(smf_cov[0,0]**0.5)
        
        smf_alphas.append(smf_val[1])
        smf_alphas_errors.append(smf_cov[1,1]**0.5)
        
        
    paper_years = [years[0], 2017]
    
    paper_Ks = [7.68e-4,7.417e-4]
    paper_Ks_errors = [0.035e-4, 0.026e-4]
    
    paper_alphas = [-1.98, -1.98]
    paper_alphas_errors = [0.005, 0.004]
    

        
    fig, axes = plt.subplots(nrows=2)#, figsize=(7, 7))
    
    axes[0].errorbar(years, pyspi_Ks, pyspi_Ks_errors, label="PySPI", capsize=2.5, linestyle='dotted')
    axes[0].errorbar(years, smf_Ks, smf_Ks_errors, label="Spimodfit", capsize=2.5, linestyle='dotted')
    axes[0].errorbar(paper_years, paper_Ks, paper_Ks_errors, label="Roques and Jourdain (2018)", capsize=2.5, linestyle='')
    
    axes[1].errorbar(years, pyspi_alphas, pyspi_alphas_errors, label="PySPI", capsize=2.5, linestyle='dotted')
    axes[1].errorbar(years, smf_alphas, smf_alphas_errors, label="Spimodfit", capsize=2.5, linestyle='dotted')
    axes[1].errorbar(paper_years, paper_alphas, paper_alphas_errors, label="Roques and Jourdain (2018)", capsize=2.5, linestyle='')
    

    
    
    axes[1].set_xlabel("Year")
    axes[0].set_ylabel("Crab K\n(Normalization at 100keV)\n[keV$^{-1}$s$^{-1}$cm$^{-2}$]")
    axes[0].ticklabel_format(style='sci', axis="y",scilimits=(1,4))
    axes[1].set_ylabel("Crab $\\alpha$")
    
    lgd = axes[0].legend(bbox_to_anchor=(1.0, 1.45), loc=1, borderaxespad=0.)
    
    plt.savefig(f"{plot_path}/crab_c_band.pdf", bbox_inches='tight', bbox_extra_artists=(lgd,))



crab_low_energy_powerlaw_combined_plot(fit_dict=combined_fits_med_pulsar_sm)
# crab_low_energy_powerlaw_combined_plot()
# crab_broken_powerlaw_100_combined_plot()
# crab_c_band_combined_plot()

combined_fits_all = {
    "0043_4_5": ["0043", "0044", "0045"],
    "0422": ["0422"],
    "0665_6": ["0665", "0666"],
    "0966_7_70": ["0966", "0967", "0970"],
    "1268_9_78": ["1268", "1269", "1278"],
    "1327_8": ["1327", "1328"],
    "1461_2_6_8": ["1461", "1462", "1466", "1468"],
    "1515_6_20_8": ["1515", "1516", "1520", "1528"],
    "1593_7_8_9": ["1593", "1597", "1598", "1599"],
    "1657_8_61_2_4_7": ["1657", "1658", "1661", "1662", "1664", "1667"],
    "1781_4_5_9": ["1781", "1784", "1785", "1789"],
    "1850_6_7": ["1850", "1856", "1857"],
    "1921_5_7_30": ["1921", "1925", "1927", "1930"],
    "1996_9_2000": ["1996", "1999", "2000"],
    "2058_62_3_6": ["2058", "2062", "2063", "2066"],
}

def print_dates():
    years = []
    for folder, revolutions in combined_fits_all.items():
        with fits.open(f"{data_path}/{revolutions[0]}/pointing.fits") as file:
            t = Table.read(file[1])
            time_start = np.array(t["TSTART"]) + 2451544.5
            time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
        years.append(time_start[-1])
    for i, folder in enumerate(combined_fits_all.keys()):
        print(folder)
        print(years[i])
        print()

# print_dates()

