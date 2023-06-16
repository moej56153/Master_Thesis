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
import pickle
from MultinestClusterFit import powerlaw_binned_spectrum

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
ra, dec = 10, -40
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





pl = Powerlaw()
pl.index = -8
pl.K = 8e-4
pl.piv = 100.
pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e0)
pl.index.prior = Uniform_prior(lower_bound=-12, upper_bound=0)
pl.index.min_value = -12.5
component1 = SpectralComponent("pl", shape=pl)
ps = PointSource("Crab", ra=ra, dec=dec, components=[component1])



spec = ps(ein)

# spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
spec_binned = powerlaw_binned_spectrum(ein, spec)

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



#optional load data from other test
path = "./main_files/large_index_error_test/bkg_20_pl_source_i_8_K_8__4"
index1 = target_scw_indices[0]
with fits.open(f"{path}/evts_det_spec.fits") as file:
    t = Table.read(file[1])

    data1 = np.zeros((len(dets), len(ebounds)-1))
    for i, d in enumerate(dets):
        data1[i] = t["COUNTS"][index1*85 + d]
        
index2 = target_scw_indices[1]
with fits.open(f"{path}/evts_det_spec.fits") as file:
    t = Table.read(file[1])

    data2 = np.zeros((len(dets), len(ebounds)-1))
    for i, d in enumerate(dets):
        data2[i] = t["COUNTS"][index2*85 + d]



# new matricies for fit

data_total = np.append(data1, data2, axis=0)
ebounds, binned_counts = log_binning_function_for_x_number_of_bins(125)(ebounds, data_total, (None, None))
# ebounds, binned_counts = no_rebinning(ebounds, data_total, (50., 2000.))

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







@njit
def bmaxba(m1, m2, t1, t2, C1, C2):
    first = C1+C2-(m1+m2)*(t1+t2)
    root = (C1+C2+(m1-m2)*(t1+t2))**2-4*C1*(m1-m2)*(t1+t2)
    res = (first+np.sqrt(root))/(2*(t1+t2))
    if res < 0.:
        return 0
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


matrices_old = (matrices1.copy(), matrices2.copy())
data_old = (data1.copy(), data2.copy())

dets_old = dets

# optional load matrices from other test
with open(f"{path}/resp_mats.pickle", "rb") as f:
    (
        pointings,
        dets,
        resp_mats,
        num_sources,
        t_elapsed,
        counts,
    ) = pickle.load(f)
    
matrices1 = resp_mats[0][0][0]
matrices2 = resp_mats[0][0][1]

# print(t_elapsed)
# print(counts)
t1, t2 = t_elapsed[0][0][0], t_elapsed[0][1][0]
data1 = counts[0][0]
data2 = counts[0][1]

# print(t1, t2)
# print(data1)
# print(data1.shape)
# print(np.array_equal(data1, data_old[0]))


# dets = dets_old
@njit
def logLcore(spec_binned):
    logL=0
    for j in range(len(dets_old)):##############################################################
        m1 = np.dot(spec_binned, matrices1[j])
        m2 = np.dot(spec_binned, matrices2[j])
        for i in range(len(m1)):
            bm = bmaxba(m1[i], m2[i], t1, t2, data1[j, i], data2[j, i])
            logL += (data1[j,i]*math.log(t1*(m1[i]+bm))+
                    data2[j,i]*math.log(t2*(m2[i]+bm))-
                    t1*(m1[i]+bm)-
                    t2*(m2[i]+bm))
        # return logL
    return logL


# functions from other test
@njit
def b_maxL_2(m, t, C):
    first = C[0]+C[1]-(m[0]+m[1])*(t[0]+t[1])
    root = (C[0]+C[1]+(m[0]-m[1])*(t[0]+t[1]))**2-4*C[0]*(m[0]-m[1])*(t[0]+t[1])
    res = (first+np.sqrt(root))/(2*(t[0]+t[1]))
    # if res < 0:
    #     return 0
    return res

@njit
def logLcore2(
    spec_binned,
    pointings,
    dets,
    resp_mats,
    num_sources,
    t_elapsed,
    counts
):
    logL=0
    for p_i in range(len(pointings)):
        for d_i in range(len(dets[p_i])):
            n_p = len(pointings[p_i])
            m = np.zeros((n_p, len(resp_mats[p_i][0][0][0,0,:])))
            
            t_b = np.zeros(n_p)
            for t_i in range(n_p):
                t_b[t_i] = t_elapsed[p_i][t_i][d_i]
            C_b = np.zeros(n_p)
            
            for s_i in range(num_sources):
                for m_i in range(n_p):
                    m[m_i,:] += np.dot(spec_binned[s_i,:], resp_mats[p_i][s_i][m_i][d_i])
            for e_i in range(len(m[0])):
                m_b = m[:,e_i]
                for C_i in range(n_p):
                    C_b[C_i] = counts[p_i][C_i][d_i, e_i]
                    
                if n_p == 2:
                    b = b_maxL_2(m_b, t_b, C_b)
                # elif n_p == 3:
                #     b = b_maxL_3(m_b, t_b, C_b)
                # elif n_p == 4:
                #     b = b_maxL_4(m_b, t_b, C_b)
                else:
                    print()
                    print("b_maxL is not defined")
                    print()
                    return 0.
                for m_i in range(n_p):
                    logL += (counts[p_i][m_i][d_i, e_i]*math.log(t_elapsed[p_i][m_i][d_i]*(m[m_i,e_i]+b))
                            -t_elapsed[p_i][0][d_i]*(m[m_i,e_i]+b))
            # return logL
    return logL




def logLba_mult(trial_values, ndim=None, params=None):
    pl.index = trial_values[1]
    pl.K = trial_values[0]
    spec = pl(ein)
    # spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
    spec_binned = powerlaw_binned_spectrum(ein, spec)
    # return logLcore(spec_binned)
    return logLcore2(
        spec_binned[np.newaxis,:],
        pointings,
        dets,
        resp_mats,
        num_sources,
        t_elapsed,
        counts
    )
    
# spec = pl(ein) 
# spec_binned = powerlaw_binned_spectrum(ein, spec)
# L1 = logLcore(spec_binned)
# L2 = logLcore2(
#         spec_binned[np.newaxis,:],
#         pointings,
#         dets,
#         resp_mats,
#         num_sources,
#         t_elapsed,
#         counts
#     )
# print(L1)
# print(L2)

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
                    logLba_mult, prior, 2, 2, n_live_points=800, resume=False, verbose=True
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
c.add_chain(chain, parameters=['K', "index", '$z$'], name='fit')

c.plotter.plot(filename="fit_corner.pdf", 
                parameters=['K', "index"],
                truth=[8e-4, -8],)

plt.savefig("./main_files/large_index_error_test/0374_02_03_test_large_index.pdf")