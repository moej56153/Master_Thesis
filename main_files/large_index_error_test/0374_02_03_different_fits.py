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
from RebinningFunctions import *
import pickle
from MultinestClusterFit import powerlaw_binned_spectrum
import pymultinest
from chainconsumer import ChainConsumer

def large_index_125_50():
    num_e_in = 50
    num_e_out = 125

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
    ein = np.geomspace(10,3000,num_e_in)
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

    # new matricies for fit

    data_total = np.append(data1, data2, axis=0)
    ebounds, binned_counts = log_binning_function_for_x_number_of_bins(num_e_out)(ebounds, data_total, (None, None))
    # ebounds, binned_counts = no_rebinning(ebounds, data_total, (None, 40.))

    data1 = binned_counts[:len(dets)]
    data2 = binned_counts[len(dets):]



    ein = np.geomspace(18, 3000, num_e_in)
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
        # if res < 0.:
        #     return 0
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


    # dets = dets_old
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
            # return logL
        return logL



    def logLba_mult(trial_values, ndim=None, params=None):
        pl.index = trial_values[1]
        pl.K = trial_values[0]
        spec = pl(ein)
        # spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
        spec_binned = powerlaw_binned_spectrum(ein, spec)
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
    c.add_chain(chain, parameters=['K', "index", '$z$'], name='fit')

    c.plotter.plot(filename="fit_corner.pdf", 
                    parameters=['K', "index"],
                    truth=[8e-4, -8],)
    
    

    plt.savefig("./main_files/large_index_error_test/0374_02_03_large_index_125_50.pdf")
    
    p = ["K", "index"]
    val = np.array([i[1] for i in c.analysis.get_summary(parameters=p).values()])
    cov = c.analysis.get_covariance(parameters=p)[1]

    with open("./main_files/large_index_error_test/source_parameters_large_index_125_50.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def large_index_500_200():
    num_e_in = 200
    num_e_out = 500

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
    ein = np.geomspace(10,3000,num_e_in)
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

    # new matricies for fit

    data_total = np.append(data1, data2, axis=0)
    ebounds, binned_counts = log_binning_function_for_x_number_of_bins(num_e_out)(ebounds, data_total, (None, None))
    # ebounds, binned_counts = no_rebinning(ebounds, data_total, (None, 40.))

    data1 = binned_counts[:len(dets)]
    data2 = binned_counts[len(dets):]



    ein = np.geomspace(18, 3000, num_e_in)
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
        # if res < 0.:
        #     return 0
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


    # dets = dets_old
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
            # return logL
        return logL



    def logLba_mult(trial_values, ndim=None, params=None):
        pl.index = trial_values[1]
        pl.K = trial_values[0]
        spec = pl(ein)
        # spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
        spec_binned = powerlaw_binned_spectrum(ein, spec)
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
    c.add_chain(chain, parameters=['K', "index", '$z$'], name='fit')

    c.plotter.plot(filename="fit_corner.pdf", 
                    parameters=['K', "index"],
                    truth=[8e-4, -8],)
    
    

    plt.savefig("./main_files/large_index_error_test/0374_02_03_large_index_500_200.pdf")
    
    p = ["K", "index"]
    val = np.array([i[1] for i in c.analysis.get_summary(parameters=p).values()])
    cov = c.analysis.get_covariance(parameters=p)[1]

    with open("./main_files/large_index_error_test/source_parameters_large_index_500_200.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
        
def large_index_1000_600():
    num_e_in = 600
    num_e_out = 1000

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
    ein = np.geomspace(10,3000,num_e_in)
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

    # new matricies for fit

    data_total = np.append(data1, data2, axis=0)
    ebounds, binned_counts = log_binning_function_for_x_number_of_bins(num_e_out)(ebounds, data_total, (None, None))
    # ebounds, binned_counts = no_rebinning(ebounds, data_total, (None, 40.))

    data1 = binned_counts[:len(dets)]
    data2 = binned_counts[len(dets):]



    ein = np.geomspace(18, 3000, num_e_in)
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
        # if res < 0.:
        #     return 0
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


    # dets = dets_old
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
            # return logL
        return logL



    def logLba_mult(trial_values, ndim=None, params=None):
        pl.index = trial_values[1]
        pl.K = trial_values[0]
        spec = pl(ein)
        # spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
        spec_binned = powerlaw_binned_spectrum(ein, spec)
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
    c.add_chain(chain, parameters=['K', "index", '$z$'], name='fit')

    c.plotter.plot(filename="fit_corner.pdf", 
                    parameters=['K', "index"],
                    truth=[8e-4, -8],)
    
    

    plt.savefig("./main_files/large_index_error_test/0374_02_03_large_index_1000_600.pdf")
    
    p = ["K", "index"]
    val = np.array([i[1] for i in c.analysis.get_summary(parameters=p).values()])
    cov = c.analysis.get_covariance(parameters=p)[1]

    with open("./main_files/large_index_error_test/source_parameters_large_index_1000_600.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def normal_index_125_50():
    num_e_in = 50
    num_e_out = 125

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
    ein = np.geomspace(10,3000,num_e_in)
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
    pl.index = -2
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

    # new matricies for fit

    data_total = np.append(data1, data2, axis=0)
    ebounds, binned_counts = log_binning_function_for_x_number_of_bins(num_e_out)(ebounds, data_total, (None, None))
    # ebounds, binned_counts = no_rebinning(ebounds, data_total, (None, 40.))

    data1 = binned_counts[:len(dets)]
    data2 = binned_counts[len(dets):]



    ein = np.geomspace(18, 3000, num_e_in)
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
        # if res < 0.:
        #     return 0
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


    # dets = dets_old
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
            # return logL
        return logL



    def logLba_mult(trial_values, ndim=None, params=None):
        pl.index = trial_values[1]
        pl.K = trial_values[0]
        spec = pl(ein)
        # spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
        spec_binned = powerlaw_binned_spectrum(ein, spec)
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
    c.add_chain(chain, parameters=['K', "index", '$z$'], name='fit')

    c.plotter.plot(filename="fit_corner.pdf", 
                    parameters=['K', "index"],
                    truth=[8e-4, -2],)
    
    

    plt.savefig("./main_files/large_index_error_test/0374_02_03_normal_index_125_50.pdf")
    
    p = ["K", "index"]
    val = np.array([i[1] for i in c.analysis.get_summary(parameters=p).values()])
    cov = c.analysis.get_covariance(parameters=p)[1]

    with open("./main_files/large_index_error_test/source_parameters_normal_index_125_50.pickle", "wb") as f:
        pickle.dump((val, cov), f)

def normal_index_125_75():
    num_e_in = 125
    num_e_out = 75

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
    ein = np.geomspace(10,3000,num_e_in)
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
    pl.index = -2
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

    # new matricies for fit

    data_total = np.append(data1, data2, axis=0)
    ebounds, binned_counts = log_binning_function_for_x_number_of_bins(num_e_out)(ebounds, data_total, (None, None))
    # ebounds, binned_counts = no_rebinning(ebounds, data_total, (None, 40.))

    data1 = binned_counts[:len(dets)]
    data2 = binned_counts[len(dets):]



    ein = np.geomspace(18, 3000, num_e_in)
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
        # if res < 0.:
        #     return 0
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


    # dets = dets_old
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
            # return logL
        return logL



    def logLba_mult(trial_values, ndim=None, params=None):
        pl.index = trial_values[1]
        pl.K = trial_values[0]
        spec = pl(ein)
        # spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
        spec_binned = powerlaw_binned_spectrum(ein, spec)
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
    c.add_chain(chain, parameters=['K', "index", '$z$'], name='fit')

    c.plotter.plot(filename="fit_corner.pdf", 
                    parameters=['K', "index"],
                    truth=[8e-4, -2],)
    
    

    plt.savefig("./main_files/large_index_error_test/0374_02_03_normal_index_125_75.pdf")
    
    p = ["K", "index"]
    val = np.array([i[1] for i in c.analysis.get_summary(parameters=p).values()])
    cov = c.analysis.get_covariance(parameters=p)[1]

    with open("./main_files/large_index_error_test/source_parameters_normal_index_125_75.pickle", "wb") as f:
        pickle.dump((val, cov), f)   

     
def normal_index_1000_400():
    num_e_in = 400
    num_e_out = 1000

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
    ein = np.geomspace(10,3000,num_e_in)
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
    pl.index = -2
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

    # new matricies for fit

    data_total = np.append(data1, data2, axis=0)
    ebounds, binned_counts = log_binning_function_for_x_number_of_bins(num_e_out)(ebounds, data_total, (None, None))
    # ebounds, binned_counts = no_rebinning(ebounds, data_total, (None, 40.))

    data1 = binned_counts[:len(dets)]
    data2 = binned_counts[len(dets):]



    ein = np.geomspace(18, 3000, num_e_in)
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
        # if res < 0.:
        #     return 0
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


    # dets = dets_old
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
            # return logL
        return logL



    def logLba_mult(trial_values, ndim=None, params=None):
        pl.index = trial_values[1]
        pl.K = trial_values[0]
        spec = pl(ein)
        # spec_binned = (ein[1:]-ein[:-1])*(spec[:-1]+spec[1:])/2
        spec_binned = powerlaw_binned_spectrum(ein, spec)
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
    c.add_chain(chain, parameters=['K', "index", '$z$'], name='fit')

    c.plotter.plot(filename="fit_corner.pdf", 
                    parameters=['K', "index"],
                    truth=[8e-4, -2],)
    
    

    plt.savefig("./main_files/large_index_error_test/0374_02_03_normal_index_1000_400.pdf")
    
    p = ["K", "index"]
    val = np.array([i[1] for i in c.analysis.get_summary(parameters=p).values()])
    cov = c.analysis.get_covariance(parameters=p)[1]

    with open("./main_files/large_index_error_test/source_parameters_normal_index_1000_400.pickle", "wb") as f:
        pickle.dump((val, cov), f)        
        
