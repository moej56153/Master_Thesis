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
from astromodels import Line, Log_uniform_prior, Uniform_prior, PointSource, SpectralComponent
import numpy as np
from RebinningFunctions import *

data_path = "./main_files/SPI_data/0374"

# Pointings and Start Times
with fits.open(f"{data_path}/pointing.fits") as file:
    t = Table.read(file[1])
    
    pointings = np.array(t["PTID_SPI"])
    
    time_start = np.array(t["TSTART"]) + 2451544.5
    time_start = [at.Time(f"{i}", format="jd").datetime for i in time_start]
    time_start = np.array([datetime.strftime(i,'%y%m%d %H%M%S') for i in time_start])
    
# Energy Bins
with fits.open(f"{data_path}/energy_boundaries.fits") as file:
    t = Table.read(file[1])
    ebounds = np.append(t["E_MIN"], t["E_MAX"][-1])
    
target_scw_ids = ('037400020010', '037400030010')
target_scw_indices = []
for i, target in enumerate(target_scw_ids):
    for j, scw in enumerate(pointings):
        if target[:8] == scw[:8]:
            target_scw_indices.append(j)
            
time1 = time_start[target_scw_indices[0]]
time2 = time_start[target_scw_indices[1]]

# ebounds = np.geomspace(18,2000,125)
ein = np.geomspace(10,3000,50)
t1 = 3000
t2 = 3000
version = find_response_version(time1)
rsp_base = ResponseDataRMF.from_version(version)
dets = get_live_dets(time=time1, event_types=["single"])

sds1 = []
sds2 = []
ra, dec = 10., -40.
for d in dets:
    rsp1 = ResponseRMFGenerator.from_time(time1,  d,
                                            ebounds, ein,
                                            rsp_base)
    rsp2 = ResponseRMFGenerator.from_time(time2,  d,
                                            ebounds, ein,
                                            rsp_base)
    sd1 = SPIDRM(rsp1, ra, dec)
    sd2 = SPIDRM(rsp2, ra, dec)
    sds1.append(sd1)
    sds2.append(sd2)


ewidth = ebounds[1:] - ebounds[:-1]
bkgs = np.full((len(dets), len(ebounds)-1), 20.) * 2 * ewidth[np.newaxis,:]

# source model





s = Line()
s.a = 0
s.a.free = False
s.b = 8e-6
s.b.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e0)
component1 = SpectralComponent("line", shape=s)
ps = PointSource("Simulated_Source_0374", ra=ra, dec=dec, components=[component1])



spec = ps(ein)

spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2

# model counts
model_count_rates1 = np.array([])
model_count_rates2 = np.array([])
for sd1, sd2 in zip(sds1, sds2):
    model_count_rates1 = np.append(model_count_rates1, np.dot(spec_binned, sd1.matrix.T))
    model_count_rates2 = np.append(model_count_rates2, np.dot(spec_binned, sd2.matrix.T))
model_count_rates1 = model_count_rates1.reshape((len(dets), -1))
model_count_rates2 = model_count_rates2.reshape((len(dets), -1))





data1 = np.array([])
data2 = np.array([])
for bkg, m1, m2 in zip(bkgs, model_count_rates1, model_count_rates2):
    data1 = np.append(data1, np.random.poisson(t1*m1 + bkg))
    data2 = np.append(data2, np.random.poisson(t2*m2 + bkg))
    
data1 = data1.reshape((len(dets), -1))
data2 = data2.reshape((len(dets), -1))


# new matricies for fit

data_total = np.append(data1, data2, axis=0)
# ebounds, binned_counts = log_binning_function_for_x_number_of_bins(125)(ebounds, data_total, (50., 2000.))
ebounds, binned_counts = no_rebinning(ebounds, data_total, (50., 2000.))

data1 = binned_counts[:len(dets)]
data2 = binned_counts[len(dets):]

ein = np.geomspace(18, 3000, 50)
sds1 = []
sds2 = []
for d in dets:
    rsp1 = ResponseRMFGenerator.from_time(time1,  d,
                                            ebounds, ein,
                                            rsp_base)
    rsp2 = ResponseRMFGenerator.from_time(time2,  d,
                                            ebounds, ein,
                                            rsp_base)
    sd1 = SPIDRM(rsp1, ra, dec)
    sd2 = SPIDRM(rsp2, ra, dec)
    sds1.append(sd1)
    sds2.append(sd2)


ewidth = ebounds[1:] - ebounds[:-1]
bkgs = np.full((len(dets), len(ebounds)-1), 20.) * 2 * ewidth[np.newaxis,:]




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
            bm = bmaxba(m1[i], m2[i], t1, t2, data1[j, i], data2[j, i])
            logL += (data1[j,i]*math.log(t1*(m1[i]+bm))+
                    data2[j,i]*math.log(t2*(m2[i]+bm))-
                    t1*(m1[i]+bm)-
                    t2*(m2[i]+bm))
    return logL

def logLba_mult(trial_values, ndim=None, params=None):
    # pl.index = trial_values[1]
    s.b = trial_values[0]
    spec = s(ein)
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
            
import pymultinest
import os
if not os.path.exists("./chains"):
    os.mkdir("chains")
sampler = pymultinest.run(
                    logLba_mult, prior, 1, 1, n_live_points=800, resume=False, verbose=True
                )

from chainconsumer import ChainConsumer
import numpy as np


def loadtxt2d(intext):
        try:
            return np.loadtxt(intext, ndmin=2)
        except:
            return np.loadtxt(intext)

c = ChainConsumer()

chain = loadtxt2d('./chains/1-post_equal_weights.dat')

#c.add_chain(chain, parameters=['K', 'index', 'F', 'mu','sigma', '$z$'], name='yeah')
c.add_chain(chain, parameters=['b', '$z$'], name='fit')

c.plotter.plot(filename="fit_corner.pdf", 
                parameters=['b'],
                truth=[8e-6],
                log_scales=[])

plt.savefig("./main_files/energy_range_error_test/simple_fit.pdf")