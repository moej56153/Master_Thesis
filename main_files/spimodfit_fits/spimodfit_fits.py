from threeML import *
import matplotlib.pyplot as plt
import numpy as np
import pickle

path = "/home/moej56153/Master_Thesis/main_files/spimodfit_fits"

def powerlaw(piv=100):
    spec = Powerlaw()
    ps = PointSource('s',l=0,b=0,spectral_shape=spec)
    ps_model = Model(ps)
    ps_model.s.spectrum.main.Powerlaw.piv = piv
    
    ps_model.s.spectrum.main.Powerlaw.K.prior = Log_uniform_prior(lower_bound=1e-15, upper_bound=1e0)
    ps_model.s.spectrum.main.Powerlaw.index.prior = Uniform_prior(lower_bound=-8, upper_bound=0)
    
    return ps_model

def broken_powerlaw(piv=100):
    spec = SmoothlyBrokenPowerLaw()
    ps = PointSource('s',l=0,b=0,spectral_shape=spec)
    ps_model = Model(ps)
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.pivot = piv
    
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.alpha.min_value = -3
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.beta.max_value = -1
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.K.max_value = 1
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.break_energy.max_value=1000
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.break_energy.min_value=1
    
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.break_scale.free = True
    
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.K.prior = Log_uniform_prior(lower_bound=1e-5, upper_bound=1e-2)
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.alpha.prior = Uniform_prior(lower_bound=-2.5, upper_bound=-1.5)
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.beta.prior = Uniform_prior(lower_bound=-5.0, upper_bound=-1.5)
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.break_energy.prior = Log_uniform_prior(lower_bound=1, upper_bound=500)
    ps_model.s.spectrum.main.SmoothlyBrokenPowerLaw.break_scale.prior = Uniform_prior(lower_bound=0.0, upper_bound=1.5)
    
    return ps_model


def data_crab(e_range, folder):
    d = OGIPLike("crab",
                observation=f'{folder}/spectra_Crab_Nebula.fits',
                response=f'{folder}/spectral_response.rmf.fits')
    
    d.set_active_measurements(e_range)
    ps_data = DataList(d)
    return ps_data
    
def data_pulsar(e_range, folder):
    d = OGIPLike("crab",
                observation=f'{folder}/spectra_A0535+26a.fits',
                response=f'{folder}/spectral_response.rmf.fits')
    
    d.set_active_measurements(e_range)
    ps_data = DataList(d)
    return ps_data

def fit_data_model(data, model):
    ps_jl = JointLikelihood(model, data)
    best_fit_parameters_ps, likelihood_values_ps = ps_jl.fit()
    ps_jl.restore_best_fit()
    
    val = np.array(best_fit_parameters_ps["value"])
    err = np.array(best_fit_parameters_ps["error"])
    cor = ps_jl.correlation_matrix
    cov = cor * err[:, np.newaxis] * err[np.newaxis, :]
    
    return val, cov, ps_jl

def bayes_analysis(data, model):
    bayes_analysis = BayesianAnalysis(model, data)
    bayes_analysis.set_sampler("multinest")
    bayes_analysis.sampler.setup(n_live_points=800, resume=False, auto_clean=True)
    bayes_analysis.sample()
    
    return bayes_analysis.results._values, bayes_analysis.results.estimate_covariance_matrix()

def save_results(val, cov, name, folder):
    with open(f"{folder}/{name}.pickle", "wb") as f:
        pickle.dump((val, cov),f)
        
def low_energy_pl(folder):
    piv = 100
    e_range="30-81.5"
    model = powerlaw(piv)
    data = data_crab(e_range, folder)
    val, cov = bayes_analysis(data, model)
    save_results(val, cov, "crab_low_energy_pl_fit", folder)
    
def brk_pl(folder):
    piv = 100
    e_range="30-400"
    model = broken_powerlaw(piv)
    data = data_crab(e_range, folder)
    val, cov = bayes_analysis(data, model)
    save_results(val, cov, "crab_brk_pl_fit", folder)
    
def pulsar_pl(folder):
    piv = 100
    e_range="30-400"
    model = powerlaw(piv)
    data = data_pulsar(e_range, folder)
    val, cov = bayes_analysis(data, model)
    save_results(val, cov, "pulsar_pl_fit", folder)
    

folders = [
    "0043_4_5",
    "0422",
    "0665_6",
    "0966_7_70",
    "1268_9_78",
    "1327_8",
    "1461_2_6_8",
    "1515_6_20_8",
    "1593_7_8_9",
    "1657_8_61_2_4_7",
    "1781_4_5_9",
    "1850_6_7",
    "1921_5_7_30",
    "1996_9_2000",
    "2058_62_3_6",
    "2126_30_4_5_7",
    "2210_2_4_5_7"
]

for folder in folders:
    l_path = f"{path}/{folder}"

    low_energy_pl(l_path)
    brk_pl(l_path)
    pulsar_pl(l_path)