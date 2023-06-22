import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
import numpy as np
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
import astropy.io.fits as fits
from astropy.table import Table
import astropy.time as at
from datetime import datetime
import matplotlib.pyplot as plt
import math
from numba import njit
from astromodels import Line, Powerlaw, Log_uniform_prior, Uniform_prior, PointSource, SpectralComponent
import numpy as np
from RebinningFunctions import *
from MultinestClusterFit import powerlaw_binned_spectrum
import pickle
from chainconsumer import ChainConsumer
import pymultinest

data_path = "./main_files/SPI_data/0374"
path = "./main_files/pure_simulation_tests"

# Pointings and Start Times
with fits.open(f"{data_path}/pointing.fits") as file:
    t = Table.read(file[1])
    
    pointings = np.array(t["PTID_SPI"])
    
    time_start = np.array(t["TSTART"]) + 2451544.5
    time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
    time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
    
    ra_spis = np.array(t["RA_SPIX"])
    dec_spis = np.array(t["DEC_SPIX"])
    
# Energy Bins
with fits.open(f"{data_path}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    ebounds = np.append(t["E_MIN"], t["E_MAX"][-1])
    
with fits.open(f"{data_path}/dead_time.fits") as file:
        t = Table.read(file[1])
        time_elapsed = np.array(t["LIVETIME"])
        
with fits.open(f"{data_path}/evts_det_spec_orig.fits") as file:
    t = Table.read(file[1])
    background_counts = np.array(t["COUNTS"])
    
target_scw_ids = ('037400020010',)
target_scw_indices = []
for i, target in enumerate(target_scw_ids):
    for j, scw in enumerate(pointings):
        if target[:8] == scw[:8]:
            target_scw_indices.append(j)
            
time = time_start[target_scw_indices[0]]
ra_spi = ra_spis[target_scw_indices[0]]
dec_spi = dec_spis[target_scw_indices[0]]



ein = np.geomspace(18,3000,75)
version = find_response_version(time)
rsp_base = ResponseDataRMF.from_version(version)
dets = get_live_dets(time=time, event_types=["single"])

detector_indices = [target_scw_indices[0]*85 + i for i in dets]
time_elapsed = time_elapsed[detector_indices]



source_piv = 200.
source_K = 1e-4
source_index = -2


ebounds, background_counts = log_binning_function_for_x_number_of_bins(125)(ebounds, background_counts, (None, None))
background_rates = background_counts[detector_indices] / time_elapsed[:,np.newaxis]

pl = Powerlaw()
pl.index = source_index
pl.K = source_K
pl.piv = source_piv

pl.K.prior = Log_uniform_prior(lower_bound=1e-8, upper_bound=1e0)
pl.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)

component1 = SpectralComponent('pl',shape=pl)
ps = PointSource('plline',ra=ra_spi, dec=dec_spi, components=[component1])
spec = ps(ein)
spec_binned = powerlaw_binned_spectrum(ein, spec)

ra1, dec1 = ra_spi, dec_spi + 2
ra2, dec2 = ra_spi, dec_spi - 2

sds1 = []
sds2 = []
for d in dets:
    rsp = ResponseRMFGenerator.from_time(time,  d,
                                            ebounds, ein,
                                            rsp_base)
    sd1 = SPIDRM(rsp, ra1, dec1)
    sd2 = SPIDRM(rsp.clone(), ra2, dec2)
    sds1.append(sd1)
    sds2.append(sd2)
    
model_count_rates1 = np.array([])
model_count_rates2 = np.array([])
for sd1, sd2 in zip(sds1, sds2):
    model_count_rates1 = np.append(model_count_rates1, np.dot(spec_binned, sd1.matrix.T))
    model_count_rates2 = np.append(model_count_rates2, np.dot(spec_binned, sd2.matrix.T))
model_count_rates1 = model_count_rates1.reshape((len(dets), -1))
model_count_rates2 = model_count_rates2.reshape((len(dets), -1))

def time_differences():
    l_path = f"{path}/time_ratio"
    
    if not os.path.exists(f"{l_path}"):
        os.mkdir(f"{l_path}")
    
    
    desired_times = ((3000,3000), (3000,4500), (3000,6000), (3000,9000), (3000,12000))
    
    with open(f"{l_path}/source_params.pickle", "wb") as f:
        pickle.dump((source_piv, source_K, source_index, desired_times), f)
    
    for i in range(len(desired_times)):
        ll_path = f"{l_path}/{i}"
        
        
        if not os.path.exists(f"{ll_path}"):
            os.mkdir(f"{ll_path}")
        
        scale_ratio1 = desired_times[i][0] / np.amax(time_elapsed)
        scale_ratio2 = desired_times[i][1] / np.amax(time_elapsed)
        time_elapsed1 = time_elapsed * scale_ratio1
        time_elapsed2 = time_elapsed * scale_ratio2
        
        data1 = np.array([])
        data2 = np.array([])
        for bkg, m1, m2, t1, t2 in zip(background_rates, model_count_rates1, model_count_rates2, time_elapsed1, time_elapsed2):
            data1 = np.append(data1, np.random.poisson((bkg + m1)*t1))
            data2 = np.append(data2, np.random.poisson((bkg + m2)*t2))
            
        data1 = data1.reshape((len(dets), -1))
        data2 = data2.reshape((len(dets), -1))
        
        @njit
        def bmaxba(m1, m2, t1, t2, C1, C2):
            first = C1+C2-(m1+m2)*(t1+t2)
            root = (C1+C2+(m1-m2)*(t1+t2))**2-4*C1*(m1-m2)*(t1+t2)
            res = (first+np.sqrt(root))/(2*(t1+t2))
            return res
        matrices1 = np.array([])
        sd_x, sd_y = sds1[0].matrix.T.shape
        for sd in sds1:
            matrices1=np.append(matrices1,sd.matrix.T)
        matrices1=matrices1.reshape((len(dets), sd_x, sd_y))

        matrices2 = np.array([])
        for sd in sds2:
            matrices2=np.append(matrices2,sd.matrix.T)
        matrices2=matrices2.reshape((len(dets), sd_x, sd_y))

        @njit
        def logLcore(spec_binned):
            logL=0
            for j in range(len(dets)):
                m1 = np.dot(spec_binned, matrices1[j])
                m2 = np.dot(spec_binned, matrices2[j])
                for i in range(len(m1)):
                    bm = bmaxba(m1[i], m2[i], time_elapsed1[j], time_elapsed2[j], data1[j, i], data2[j, i])
                    logL += (data1[j,i]*math.log(time_elapsed1[j]*(m1[i]+bm))+
                            data2[j,i]*math.log(time_elapsed2[j]*(m2[i]+bm))-
                            time_elapsed1[j]*(m1[i]+bm)-
                            time_elapsed2[j]*(m2[i]+bm))
            return logL

        def logLba_mult(trial_values, ndim=None, params=None):
            pl.index = trial_values[1]
            pl.K = trial_values[0]
            spec = pl(ein)
            spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
            return logLcore(spec_binned)
        
        def prior(params, ndim=None, nparams=None):
            for i, (parameter_name, parameter) in enumerate(
                ps.free_parameters.items()
            ):

                try:

                    params[i] = parameter.prior.from_unit_cube(params[i])

                except AttributeError:

                    raise RuntimeError(
                        "The prior you are trying to use for parameter %s is "
                        "not compatible with sampling from a unitcube"
                        % parameter_name
                    )
                    
        if not os.path.exists("./chains"):
            os.mkdir("chains")
        sampler = pymultinest.run(
                            logLba_mult, prior, 2, 2, n_live_points=800, resume=False, verbose=True
                        )
        def loadtxt2d(intext):
            try:
                return np.loadtxt(intext, ndmin=2)
            except:
                return np.loadtxt(intext)

        c = ChainConsumer()

        chain = loadtxt2d('./chains/1-post_equal_weights.dat')

        #c.add_chain(chain, parameters=['K', 'index', 'F', 'mu','sigma', '$z$'], name='yeah')
        c.add_chain(chain, parameters=['K', 'index', '$z$'], name='fit')
        
        p = ["K", "index"]
        val = np.array([i[1] for i in c.analysis.get_summary(parameters=p).values()])
        cov = c.analysis.get_covariance(parameters=p)[1]

        with open(f"{ll_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)
            
        c.plotter.plot(filename="fit_corner.pdf", 
                parameters=['K', "index"],
                truth=[source_K, source_index],)
        plt.savefig(f"{ll_path}/fit_parameters.pdf")
        
time_differences()