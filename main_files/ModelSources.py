import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from astromodels import Powerlaw, SmoothlyBrokenPowerLaw, Line, Log_uniform_prior, Uniform_prior, PointSource, SpectralComponent, Model
from CustomAstromodels import C_Band

def define_sources(source_funcs):    
    model = Model()
    for source_func, params in source_funcs:
        source_func(model, *params)
    return model

def crab_pl_fixed_pos(model, piv):
    ra, dec = 83.6333, 22.0144
    
    pl = Powerlaw()
    pl.piv = piv
    pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e0)
    pl.index.prior = Uniform_prior(lower_bound=-3, upper_bound=-1)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("Crab", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def crab_pl_free_pos(model, piv):
    ra, dec = 83.6333, 22.0144
    angle_range = 5.
    
    pl = Powerlaw()
    pl.piv = piv
    pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e0)
    pl.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("Crab", ra=ra, dec=dec, components=[component1])
    ps.position.ra.free = True
    ps.position.ra.prior = Uniform_prior(
        lower_bound = ra - abs(angle_range/np.cos(dec)),
        upper_bound = ra + abs(angle_range/np.cos(dec))
    )
    ps.position.dec.free = True
    ps.position.dec.prior = Uniform_prior(
        lower_bound = dec - angle_range,
        upper_bound = dec + angle_range
    )
    
    model.add_source(ps)
    return model

def crab_sm_br_pl(model, piv=100):
    ra, dec = 83.6333, 22.0144
    
    pl = SmoothlyBrokenPowerLaw()
    pl.piv = piv
    pl.alpha.min_value = -2.5
    pl.K.prior = Log_uniform_prior(lower_bound=5e-4, upper_bound=1e-3)
    pl.alpha.prior = Uniform_prior(lower_bound=-2.2, upper_bound=-1.8)
    pl.beta.prior = Uniform_prior(lower_bound=-5.0, upper_bound=-1.9)
    pl.break_energy.prior = Log_uniform_prior(lower_bound=50, upper_bound=500)
    pl.break_scale.prior = Uniform_prior(lower_bound=0.0, upper_bound=1.5)
    pl.break_scale.free = True
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("Crab", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model
    

def crab_lower_band(model, piv=100):
    ra, dec = 83.6333, 22.0144
    
    s = C_Band()
    s.piv = piv
    s.alpha.min_value = -2.2
    s.alpha = -2.
    s.beta = -2.25
    s.xp = 500
    s.K.prior = Log_uniform_prior(lower_bound=5e-4, upper_bound=1e-3)
    s.alpha.prior = Uniform_prior(lower_bound=-2.2, upper_bound=-1.8)
    s.xp.prior = Uniform_prior(lower_bound=300, upper_bound=1000)
    s.beta.free = False
    s.xp.free = False
    component1 = SpectralComponent("band", shape=s)
    ps = PointSource("Crab", ra=ra, dec=dec, components=[component1])
    model.add_source(ps)
    return model

def s_1A_0535_262_pl(model, piv):
    ra, dec = 84.7270, 26.3160
    
    pl = Powerlaw()
    pl.piv = piv
    pl.K.prior = Log_uniform_prior(lower_bound=1e-15, upper_bound=1e-3)
    pl.index.prior = Uniform_prior(lower_bound=-8, upper_bound=0)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("A_0535_262", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def s_4U_0517_17_pl(model, piv):
    ra, dec = 77.6896, 16.4986
    
    pl = Powerlaw()
    pl.piv = piv
    pl.index.min_value = -20.
    pl.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e-4)
    pl.index.prior = Uniform_prior(lower_bound=-20, upper_bound=10)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("_4U_0517__17", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def s_4U_0614_09_pl(model, piv):
    ra, dec = 94.2800, 9.13700
    
    pl = Powerlaw()
    pl.piv = piv
    pl.index.max_value = 20.
    pl.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e-4)
    pl.index.prior = Uniform_prior(lower_bound=-10, upper_bound=20)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("_4U_0614__09", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def geminga_pl(model, piv):
    ra, dec = 98.4750, 17.7670
    
    pl = Powerlaw()
    pl.piv = piv
    pl.index.max_value = 15.
    pl.K.prior = Log_uniform_prior(lower_bound=1e-15, upper_bound=1e-5)
    pl.index.prior = Uniform_prior(lower_bound=-10, upper_bound=15)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("Geminga", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def simulated_pl_0374(model, piv):
    ra, dec = 10., -40.
    
    pl = Powerlaw()
    pl.piv = piv
    pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e0)
    pl.index.prior = Uniform_prior(lower_bound=-12, upper_bound=4)
    pl.index.min_value = -12.5
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("Simulated_Source_0374", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def simulated_pl_0374_free_pos(model, piv):
    ra, dec = 10., -40.
    angle_range = 5.
    
    pl = Powerlaw()
    pl.piv = piv
    pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e0)
    pl.index.prior = Uniform_prior(lower_bound=-12, upper_bound=4)
    pl.index.min_value = -12.5
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("Simulated_Source_0374", ra=ra, dec=dec, components=[component1])
    ps.position.ra.free = True
    ps.position.ra.prior = Uniform_prior(
        lower_bound = ra - abs(angle_range/np.cos(dec)),
        upper_bound = ra + abs(angle_range/np.cos(dec))
    )
    ps.position.dec.free = True
    ps.position.dec.prior = Uniform_prior(
        lower_bound = dec - angle_range,
        upper_bound = dec + angle_range
    )
    
    model.add_source(ps)
    return model

def simulated_linear_0374(model):
    ra, dec = 10., -40.
    
    s = Line()
    s.a = 0
    s.a.free = False
    s.b.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e0)
    component1 = SpectralComponent("line", shape=s)
    ps = PointSource("Simulated_Source_0374", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def simulated_pl_1380(model, piv):
    ra, dec = 155., 75.
    
    pl = Powerlaw()
    pl.piv = piv
    pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e0)
    pl.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("Simulated_Source_1380", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def true_values(include_position=False):
    piv = 40
    
    crab_parameters = np.array([[9.3, 1, -2.08, 83.6333, 22.0144],
                                [7.52e-4, 100, -1.99, 83.6333, 22.0144],
                                [11.03, 1, -2.1, 83.6333, 22.0144]])
    
    crab_values = np.zeros((len(crab_parameters), len(crab_parameters[0]) - 1))
    crab_values[:,0] = crab_parameters[:,0] * (piv / crab_parameters[:,1])**crab_parameters[:,2]
    crab_values[:,1:] = crab_parameters[:,2:]
    
    names = ["Crab K", "Crab index", "Crab ra", "Crab dec"]
    if include_position:
        crab_values = (names, crab_values)
    else:
        crab_values = (names[:-2], crab_values[:,:-2])
    
    return crab_values