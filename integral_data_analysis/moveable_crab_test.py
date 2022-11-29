from astropy.coordinates import SkyCoord
import numpy as np
from IntegralQuery import SearchQuery, IntegralQuery, Filter, Range
from IntegralPointingClustering import ClusteredQuery
import astropy.io.fits as fits
from astropy.table import Table
from datetime import datetime
import matplotlib.pyplot as plt
import math
from numba import njit
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
from pyspi.utils.livedets import get_live_dets
from astromodels import Powerlaw, Log_uniform_prior, Uniform_prior, PointSource, SpectralComponent, Model
from chainconsumer import ChainConsumer
import pymultinest
import os
import astropy.time as at
from scipy.stats import poisson
import pickle

def extract_date_range(path):
    with fits.open(f"{path}/pointing.fits") as file:
        t = Table.read(file[1])
        t1 = at.Time(f'{t["TSTART"][0]+2451544.5}', format='jd')
        t1.format = "isot"
        t2 = at.Time(f'{t["TSTOP"][-1]+2451544.5}', format='jd')
        t2.format = "isot"
    return t1.value, t2.value

def get_scw_ids(path, radius_around_crab, min_time_elapsed, print_results=False):
    p = SkyCoord(83.6333, 22.0144, frame="icrs", unit="deg")
    searchquerry = SearchQuery(position=p, radius=f"{radius_around_crab} degree")
    cat = IntegralQuery(searchquerry)

    f = Filter(SCW_TYPE="POINTING",
               TIME=Range(*extract_date_range(path)))
    scw_ids_all = cat.apply_filter(f, return_coordinates=True, remove_duplicates=True)
    
    scw_ids = []
    
    multiple_files = []
    no_files = []
    no_pyspi = []
    
    num_dets = 19
    eb = np.geomspace(18, 2000, 5)
    emod = np.geomspace(18, 2000, 5)
    for scw_id in scw_ids_all:
        good = True
        with fits.open(f"{path}/pointing.fits") as file:
            t = Table.read(file[1])
            index = np.argwhere(t["PTID_ISOC"]==scw_id[0][:8])
            
            if len(index) < 1:
                no_files.append(scw_id)
                good = False
                continue
                
            elif len(index) > 1:
                multiple_files.append(scw_id)
                good = False
                            
            pointing_info = t[index[-1][0]]
        
            t1 = at.Time(f'{pointing_info["TSTART"]+2451544.5}', format='jd').datetime
            time_start = datetime.strftime(t1,'%y%m%d %H%M%S')
                            
            with fits.open(f"{path}/dead_time.fits") as file2:
                t2 = Table.read(file2[1])
                
                time_elapsed = np.zeros(num_dets)
                
                for i in range(num_dets):
                    for j in index:
                        time_elapsed[i] += t2["LIVETIME"][j[0]*85 + i]
                            
            dets = get_live_dets(time=time_start, event_types=["single"])
                            
            if not np.amin(time_elapsed[dets]) > min_time_elapsed:
                good = False
        
        try: # investigate why this is necessary
            version1 = find_response_version(time_start)
            rsp_base = ResponseDataRMF.from_version(version1)
            rsp1 = ResponseRMFGenerator.from_time(time_start, dets[0], eb, emod, rsp_base)
        except:
            no_pyspi.append(scw_id)
            good = False
            
        if good:
            scw_ids.append(scw_id)
            
    if print_results:
        print("Multiple Files:")
        print(multiple_files)
        print("No Files:")
        print(no_files)
        print("No PySpi:")
        print(no_pyspi)
        print("Good:")
        print(scw_ids)
    
    return np.array(scw_ids)

def create_pair_clusters_crab(
    paths,
    min_angle_dif,
    max_angle_dif,
    max_time_dif,
    radius_around_crab,
    min_time_elapsed
): # check if dets is same in pair?
    output = []
    for path in paths:
        scw_ids = get_scw_ids(path, radius_around_crab, min_time_elapsed)
        cq = ClusteredQuery(scw_ids,
                            angle_weight=0.,
                            time_weight=1./max_time_dif,
                            max_distance=1.,
                            min_ang_distance=min_angle_dif,
                            max_ang_distance=max_angle_dif,
                            cluster_size_range = (2,2),
                            failed_improvements_max = 3,
                            suboptimal_cluster_size = 1,
                            close_suboptimal_cluster_size = 1
                            ).get_clustered_scw_ids()
        for pair in cq[2]:
            output.append((path, pair[0], pair[1]))
    return tuple(output)

def save_pair_clusters(pointings, folder):
    if not os.path.exists(f"./{folder}"):
        os.mkdir(folder)
    with open(f"./{folder}/pointings_pickle", "wb") as f:
        pickle.dump(pointings, f)
        
def load_pair_clusters(folder):
    with open(f"./{folder}/pointings_pickle", "rb") as f:
        pointings = pickle.load(f)
    return pointings

def extract_pointing_info(path, p_id):
    num_dets = 19
    with fits.open(f"{path}/pointing.fits") as file:
        t = Table.read(file[1])
        index = np.argwhere(t["PTID_ISOC"]==p_id[:8])
        
        if len(index) < 1:
            raise Exception(f"{p_id} not found")

        pointing_info = t[index[-1][0]]
        
        t1 = at.Time(f'{pointing_info["TSTART"]+2451544.5}', format='jd').datetime
        time_start = datetime.strftime(t1,'%y%m%d %H%M%S')
            
    with fits.open(f"{path}/dead_time.fits") as file:
        t = Table.read(file[1])
        
        time_elapsed = np.zeros(num_dets)
        
        for i in range(num_dets):
            for j in index:
                time_elapsed[i] += t["LIVETIME"][j[0]*85 + i]
        
    with fits.open(f"{path}/energy_boundaries.fits") as file:
        t = Table.read(file[1])
        
        energy_bins = np.append(t["E_MIN"], t["E_MAX"][-1])
    
    with fits.open(f"{path}/evts_det_spec.fits") as file:
        t = Table.read(file[1])
        
        counts = np.zeros((num_dets, len(energy_bins)-1))
        for i in range(num_dets):
            for j in index:
                counts[i, : ] += t["COUNTS"][j[0]*85 + i]

    return time_start, time_elapsed, energy_bins, counts

def prepare_fit_data(
    pointings,
    binning_func,
    energy_range
):
    ebs = [] # do i need this?
    counts = []
    dets = []
    t_elapsed = []
    t_start = []
        
    for pair in pointings:
        time_start1, time_elapsed1, energy_bins1, counts_f1 = extract_pointing_info(pair[0], pair[1])
        time_start2, time_elapsed2, energy_bins2, counts_f2 = extract_pointing_info(pair[0], pair[2])
        t_start.append((time_start1, time_start2))
        
        dets1 = get_live_dets(time=time_start1, event_types=["single"])
        dets2 = get_live_dets(time=time_start2, event_types=["single"])
        assert np.array_equal(dets1, dets2), f"Active detectors are not the same for {pair[1]} and {pair[2]}"
        dets.append(dets1)
        
        t_elapsed.append((time_elapsed1[dets1], time_elapsed2[dets1]))
        
        assert np.array_equal(energy_bins1, energy_bins2), f"Energy bins are not the same for {pair[1]} and {pair[2]}"
        
        eb, c = binning_func(
            energy_bins1,
            np.append(counts_f1[dets1], counts_f2[dets1], axis=0),
            energy_range)
        counts.append((c[:len(dets1)], c[len(dets1):]))
        ebs.append(eb)
            
    ebs = tuple(ebs) 
    counts = tuple(counts)
    dets = tuple(dets)
    t_elapsed = tuple(t_elapsed)
    t_start = tuple(t_start)
    
    return ebs, counts, dets, t_elapsed, t_start

def initialize_resp_mats(
    pointings,
    ebs,
    emod,
    t_start,
    dets,
    source_model,
):
    # index order: tuple(combination, source, pointing, np_array(dets, e_in, e_out))
    resp_mats = []
    
    for count, combination in enumerate(pointings):
        version1 = find_response_version(t_start[count][0])
        for pointing in range(1, len(combination[1:])):
            version2 = find_response_version(t_start[count][pointing])
            assert version1 == version2, f"Response versions are not equal for {combination[1]} and {combination[pointing+1]}"
        
        rsp_base = ResponseDataRMF.from_version(version1)
        source_resp_mats = []
        for source in source_model.sources.values():
            combination_resp_mats = []
            for pointing in range(len(combination[1:])):
                combination_resp_mats.append(
                    generate_resp_mat(
                        t_start[count][pointing],
                        dets[count],
                        ebs[count],
                        emod,
                        source.position.get_ra(),
                        source.position.get_dec(),
                        rsp_base,
                    )
                )
            source_resp_mats.append(tuple(combination_resp_mats))
            
        #     sds1 = np.array([])
        #     sds2 = np.array([])
        #     for d in dets[count]:
        #         rsp1 = ResponseRMFGenerator.from_time(t_start[count][0], d, ebs[count], emod, rsp_base)
        #         sd1 = SPIDRM(rsp1, source[1].position.get_ra(), source[1].position.get_dec())
        #         sds1 = np.append(sds1, sd1.matrix.T)
        #         rsp2 = ResponseRMFGenerator.from_time(t_start[count][1], d, ebs[count], emod, rsp_base)
        #         sd2 = SPIDRM(rsp2, source[1].position.get_ra(), source[1].position.get_dec())
        #         sds2 = np.append(sds2, sd2.matrix.T)
        #     source_resp_mats.append((
        #         sds1.reshape((len(dets[count]), len(emod)-1, len(ebs[count])-1)),
        #         sds2.reshape((len(dets[count]), len(emod)-1, len(ebs[count])-1)),
        #     ))
        resp_mats.append(tuple(source_resp_mats))
        
    resp_mats = tuple(resp_mats)
    
    return resp_mats

def update_resp_mats(
    resp_mats,
    pointings,
    ebs,
    emod,
    t_start,
    dets,
    source_model,
    updatable_sources
):
    for count, combination in enumerate(pointings):
        version = find_response_version(t_start[count][0])
        rsp_base = ResponseDataRMF.from_version(version)
        for source_num, (source_name, source) in enumerate(source_model.sources.items()):
            if source_name in updatable_sources:
                for pointing in range(len(combination[1:])):
                    resp_mats[count][source_num][pointing][:,:,:] = generate_resp_mat(
                        t_start[count][pointing],
                        dets[count],
                        ebs[count],
                        emod,
                        source.position.get_ra(),
                        source.position.get_dec(),
                        rsp_base,
                    )

def generate_resp_mat(
    time,
    dets,
    ebs,
    emod,
    ra,
    dec,
    rsp_base,
):
    sds = np.empty(0)
    for d in dets:
        rsp = ResponseRMFGenerator.from_time(time, d, ebs, emod, rsp_base)
        sd = SPIDRM(rsp, ra, dec)
        sds = np.append(sds, sd.matrix.T)
    return sds.reshape((len(dets), len(emod)-1, len(ebs)-1))

def find_updatable_sources(source_model):
    keywords = ["position"]
    updatable_sources = []
    for parameter in source_model.free_parameters.values():
        first_pos = parameter.path.find(".")
        second_pos = parameter.path.find(".", first_pos+1)
        if parameter.path[first_pos+1 : second_pos] in keywords and not parameter.path[:first_pos] in updatable_sources:
            updatable_sources.extend([parameter.path[:first_pos]])
    return updatable_sources

def run_multinest_powerlaws(
    pointings,
    ebs,
    emod,
    counts,
    t_start,
    t_elapsed,
    dets,
    resp_mats,
    source_model,
):
    num_sources = len(source_model.sources)
    updatable_sources = find_updatable_sources(source_model)
    
    
    @njit
    def bmaxba(m1, m2, t1, t2, C1, C2):
        first = C1+C2-(m1+m2)*(t1+t2)
        root = (C1+C2+(m1-m2)*(t1+t2))**2-4*C1*(m1-m2)*(t1+t2)
        res = (first+np.sqrt(root))/(2*(t1+t2))
        if res < 0:
            return 0
        return res
    
    @njit
    def logLcore(spec_binned):
        logL=0
        for k in range(len(pointings)):
            for j in range(len(dets[k])):
                m1 = np.zeros(len(resp_mats[k][0][0][0,0,:]))
                m2 = m1.copy()
                for i in range(num_sources):
                    m1 += np.dot(spec_binned[i,:], resp_mats[k][i][0][j])
                    m2 += np.dot(spec_binned[i,:], resp_mats[k][i][1][j])
                for i in range(len(m1)):
                    bm = bmaxba(m1[i], m2[i], t_elapsed[k][0][j], t_elapsed[k][1][j], counts[k][0][j, i], counts[k][1][j, i])
                    logL += (counts[k][0][j, i]*math.log(t_elapsed[k][0][j]*(m1[i]+bm))+
                            counts[k][1][j, i]*math.log(t_elapsed[k][1][j]*(m2[i]+bm))-
                            t_elapsed[k][0][j]*(m1[i]+bm)-
                            t_elapsed[k][1][j]*(m2[i]+bm))
        return logL

    def logLba_mult(trial_values, ndim=None, params=None):
        spec_binned = np.zeros((num_sources, len(emod)-1))
        for i, parameter in enumerate(source_model.free_parameters.values()):
            parameter.value = trial_values[i]
        for i, source in enumerate(source_model.sources.values()):
            spec = source(emod)
            spec_binned[i,:] = (emod[1:]-emod[:-1])*(spec[:-1]+spec[1:])/2
        if updatable_sources:
            update_resp_mats(
                resp_mats,
                pointings,
                ebs,
                emod,
                t_start,
                dets,
                source_model,
                updatable_sources,
            )
        return logLcore(spec_binned)
    
    def prior(params, ndim=None, nparams=None):
        # for j, source in enumerate(sources):
        #     for i, (parameter_name, parameter) in enumerate(
        #         source[1].free_parameters.items() ########################################## dictionary not ordered?
        #     ):
        #         try:
        #             params[i + j*2] = parameter.prior.from_unit_cube(params[i + j*2])

        #         except AttributeError:
        #             raise RuntimeError(
        #                 "The prior you are trying to use for parameter %s is "
        #                 "not compatible with sampling from a unitcube"
        #                 % parameter_name
        #             )
        for i, parameter in enumerate(source_model.free_parameters.values()):
            try:
                params[i] = parameter.prior.from_unit_cube(params[i])

            except AttributeError:
                raise RuntimeError(
                    "The prior you are trying to use for parameter %s is "
                    "not compatible with sampling from a unitcube"
                    % parameter.path
                )

    num_params = len(source_model.free_parameters)
    
    # ###
    # trial_values = [4.5e-3, -2.08, 7e-3, -2.5]
    # spec_binned = np.zeros((num_sources, len(emod)-1))
    # for i, parameter in enumerate(source_model.free_parameters.values()):
    #     parameter.value = trial_values[i]
    # for i, source in enumerate(source_model.sources.values()):
    #     spec = source(emod)
    #     spec_binned[i,:] = (emod[1:]-emod[:-1])*(spec[:-1]+spec[1:])/2
    # print(logLcore(spec_binned))
    # return None
    # ###

    if not os.path.exists("./chains"):
        os.mkdir("chains")
    sampler = pymultinest.run(
        logLba_mult, prior, num_params, num_params, n_live_points=800, resume=False, verbose=True
    )
    
def ppc(
    pointings,
    parameters,
    resp_mats,
    ebs,
    counts,
    emod,
    dets,
    t_elapsed,
    source_model,
    folder
):
    
    for i, pair in enumerate(pointings):
        s1, s2, b1, b2 = calc_rates_powerlaw(
            parameters,
            resp_mats[i],
            ebs[i],
            counts[i],
            emod,
            dets[i],
            t_elapsed[i],
            source_model
        )
        
        qq_plot(b1, s1, ebs[i], counts[i][0], dets[i], folder, pair[1])
        qq_plot(b2, s2, ebs[i], counts[i][1], dets[i], folder, pair[2])
        
def calc_fit_quality_crab(summary, cov, true_vals):
    fit_val = np.array([summary["Crab K"][1], summary["Crab index"][1]])
    fit_cov = cov[1]
    rel_distance = []
    
    for i in range(len(true_vals)):
        dif = fit_val - true_vals[i]
        
        rel_distance.append(np.sqrt(
            np.linalg.multi_dot([dif, np.linalg.inv(fit_cov), dif])
        ))
        
    return np.array(rel_distance)

def calc_rates_powerlaw(
    parameters,
    resp_mats,
    eb,
    counts,
    emod,
    dets,
    t_elapsed,
    source_model
):
    source_rate1 = np.zeros((len(dets), len(eb)-1, len(parameters)))
    source_rate2 = np.zeros((len(dets), len(eb)-1, len(parameters)))
    background_rate = np.zeros((len(dets), len(eb)-1, len(parameters)))
    
    num_sources = len(source_model.sources)
    
    @njit
    def bmaxba(m1, m2, t1, t2, C1, C2):
        first = C1+C2-(m1+m2)*(t1+t2)
        root = (C1+C2+(m1-m2)*(t1+t2))**2-4*C1*(m1-m2)*(t1+t2)
        res = (first+np.sqrt(root))/(2*(t1+t2))
        if res < 0:
            return 0
        return res
    
    for k, params in enumerate(parameters):
        spec_binned = np.zeros((num_sources, len(emod)-1))
        for i, parameter in enumerate(source_model.free_parameters.values()):
            parameter.value = params[i]
        for i, source in enumerate(source_model.sources.values()):
            spec = source(emod)
            spec_binned[i,:] = (emod[1:]-emod[:-1])*(spec[:-1]+spec[1:])/2
            
        for j in range(len(dets)):
            for i in range(num_sources):
                source_rate1[j,:,k] += np.dot(spec_binned[i,:], resp_mats[i][0][j])
                source_rate2[j,:,k] += np.dot(spec_binned[i,:], resp_mats[i][1][j])
            for i in range(len(eb)-1):
                background_rate[j,i,k] = bmaxba(
                    source_rate1[j,i,k],
                    source_rate2[j,i,k],
                    t_elapsed[0][j],
                    t_elapsed[1][j],
                    counts[0][j, i],
                    counts[1][j, i]
                )
         
    source_rate1 = np.average(source_rate1, axis=2) * t_elapsed[0][:,np.newaxis]
    source_rate2 = np.average(source_rate2, axis=2) * t_elapsed[1][:,np.newaxis]
    background_rate1 = np.average(background_rate, axis=2) * t_elapsed[0][:,np.newaxis]
    background_rate2 = np.average(background_rate, axis=2) * t_elapsed[1][:,np.newaxis]

    return source_rate1, source_rate2, background_rate1, background_rate2

def qq_plot(
    b,
    s,
    eb,
    c,
    dets,
    folder,
    name
):
    fig, axes = plt.subplots(5,4, sharex=True, sharey=True, figsize=(10,10))
    axes = axes.flatten()
    i=0
    predicted = b + s
    predicted_lower = poisson.ppf(0.05, predicted)
    predicted_upper = poisson.ppf(0.95, predicted)
    # p = b + s
    # predicted = np.cumsum(p, axis=1)
    # predicted_lower = np.cumsum(poisson.ppf(0.05, p), axis=1) # find non-integer solution?
    # predicted_upper = np.cumsum(poisson.ppf(0.95, p), axis=1)
    
    counts = c
    # counts = np.cumsum(c, axis=1)
    
    for d in range(19):
        axes[d].text(.5,.9,f"Det {d}",horizontalalignment='center',transform=axes[d].transAxes)
        if d in dets:
            line1, = axes[d].step(eb[:-1], predicted[i], c="r")
            line2, = axes[d].step(eb[:-1], counts[i], c="k")
            axes[d].fill_between(eb[:-1], predicted_lower[i], predicted_upper[i], color="r", alpha=0.5)
            if i==0:
                line1.set_label("Predicted Counts")
                line2.set_label("Real Counts")
            i += 1
        axes[d].set_yscale("log")
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.subplots_adjust(hspace=0, top=0.96, bottom=0.1)
    lgd = fig.legend(loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize='x-large')
    
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Detected Energy [keV]")
    plt.ylabel("Cumulative Counts")
    
    fig.savefig(f"{folder}/{name}.pdf")
    plt.close()
    
def extract_parameter_names_simple(model):
    parameter_names = []
    for full_name in model.free_parameters.keys():
        source = full_name[ : full_name.find(".")]
        source = source[1:] if source[0]=="_" else source
        source = source.replace("__", "+").replace("_", " ")
        parameter = full_name[-1 * full_name[::-1].find(".") : ]
        parameter_names.extend([f"{source} {parameter}"])
    return parameter_names

def multinest_for_pointing_pairs(
    folder,
    pointings,
    source_model,
    # true_values_main,
    binning_func,
    energy_range,
    emod,
):
    
    ebs, counts, dets, t_elapsed, t_start = prepare_fit_data(pointings, binning_func, energy_range)
    
    resp_mats = initialize_resp_mats(pointings, ebs, emod, t_start, dets, source_model)
    
    run_multinest_powerlaws(pointings, ebs, emod, counts, t_start, t_elapsed, dets, resp_mats, source_model)
    
    parameter_names = extract_parameter_names_simple(source_model)
    parameter_names.extend(["$z$"])
        
    c = ChainConsumer()
    chain = np.loadtxt('./chains/1-post_equal_weights.dat')
    c.add_chain(chain, parameters=parameter_names, name='fit')
    
    fig = c.plotter.plot(
        filename="crab_parameter_fit.pdf",
        parameters=parameter_names[:-1],
        # truth={'Crab K':true_values_main[0,0], 'Crab index':true_values_main[0,1]},
        figsize=1.5
    )
    
    plt.savefig(f"{folder}/parameter_fit_distributions.pdf")
    plt.close()
    
    summary = c.analysis.get_summary(parameters=parameter_names[:-1])
    # cov = c.analysis.get_covariance(parameters=['Crab K', 'Crab index'])
    
    # rel_distances = calc_fit_quality_crab(summary, cov, true_values_main)
    
    # with open(f"{folder}/reference_values", "w") as f:
    #     f.write("Crab_K Crab_index Rel._Dist.\n")
    #     for i in range(len(true_values_main)):
    #         f.write(f"{true_values_main[i,0]:.5f}   {true_values_main[i,1]:.2f}   {rel_distances[i]:.2f}\n")
            
    with open(f"{folder}/pointing_combinations", "w") as f:
        for combination in pointings:
            f.write(f'{"  ".join(combination[1:])}\n')
            
    with open(f"{folder}/parameter_fit_constraints", "w") as f:
        for param in parameter_names[:-1]:
            f.write(f"{param}:\n")
            try:
                f.write(f"{summary[param][0]:.5}  {summary[param][1]:.5}  {summary[param][2]:.5}\n")
            except:
                f.write(f"None  {summary[param][1]:.5}  None\n")
            
    ppc(
        pointings,
        chain,
        resp_mats,
        ebs,
        counts,
        emod,
        dets,
        t_elapsed,
        source_model,
        folder
    )

def rebin_data_exp(
    bins,
    counts,
    energy_range
):

    if energy_range[0]:
        for i, e in enumerate(bins):
            if e > energy_range[0]:
                bins = bins[i:]
                counts = counts[:,i:] ############should these be the same?
                break
    if energy_range[1]:
        for i, e in enumerate(bins):
            if e > energy_range[1]:
                bins = bins[:i]
                counts = counts[:,:i-1]
                assert i > 1, "Max Energy is too low"
                break
        
    min_counts = 5
    
    max_num_bins = 120
    min_num_bins = 1
    
    finished = False
    
    while not finished:
        num_bins = round((max_num_bins + min_num_bins) / 2)
        
        if num_bins == max_num_bins or num_bins == min_num_bins:
            num_bins = min_num_bins
            finished = True
        
        temp_bins = np.geomspace(bins[0], bins[-1], num_bins+1)
        
        new_bins, new_counts = rebin_closest(bins, counts, temp_bins)
        
        if np.amin(new_counts) < min_counts:
            max_num_bins = num_bins
        else:
            min_num_bins = num_bins
            
    return new_bins, new_counts
    
# @njit
def rebin_closest(bins, counts, temp_bins):
    counts = np.copy(counts)
    closest1 = len(bins) - 1
    for i in range(len(temp_bins)-2, 0, -1):
        closest2 = np.argpartition(
            np.absolute(bins - temp_bins[i]),
            0
        )[0]
        if closest1 - closest2 >= 2:
            counts[:,closest2] += np.sum(counts[:, closest2+1 : closest1], axis=1)
            counts = np.delete(
                counts,
                [j for j in range(closest2+1, closest1)],
                axis=1
            )
            bins = np.delete(
                bins,
                [j for j in range(closest2+1, closest1)]
            )
        closest1 = closest2
    return bins, counts


# counts = np.linspace(1,40,40).reshape((2,20))
# bins = np.linspace(1,21,21)
# temp_bins = np.geomspace(1,21,11)
# b, c = rebin_closest(bins, counts, temp_bins)
# print(bins)
# print(counts)
# print(np.sum(counts, axis=1))
# print(temp_bins)
# print(b)
# print(c)
# print(np.sum(c, axis=1))
    
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
    pl.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)
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

def _1A_0535_262_pl(model, piv):
    ra, dec = 84.7270, 26.3160
    
    pl = Powerlaw()
    pl.piv = piv
    pl.K.prior = Log_uniform_prior(lower_bound=1e-10, upper_bound=1e0)
    pl.index.prior = Uniform_prior(lower_bound=-4, upper_bound=0)
    component1 = SpectralComponent("pl", shape=pl)
    ps = PointSource("_1A_0535__262", ra=ra, dec=dec, components=[component1])
    
    model.add_source(ps)
    return model

def _4U_0517_17_pl(model, piv):
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

def _4U_0614_09_pl(model, piv):
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

