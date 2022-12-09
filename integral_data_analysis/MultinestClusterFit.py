# from astropy.coordinates import SkyCoord
import numpy as np
# from IntegralQuery import SearchQuery, IntegralQuery, Filter, Range
# from IntegralPointingClustering import ClusteredQuery
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
# from astromodels import Powerlaw, Log_uniform_prior, Uniform_prior, PointSource, SpectralComponent, Model
from chainconsumer import ChainConsumer
import pymultinest
import os
import astropy.time as at
from scipy.stats import poisson
# import pickle

rsp_bases = tuple([ResponseDataRMF.from_version(i) for i in range(5)])

@njit
def b_maxL_2(m, t, C):
    first = C[0]+C[1]-(m[0]+m[1])*(t[0]+t[1])
    root = (C[0]+C[1]+(m[0]-m[1])*(t[0]+t[1]))**2-4*C[0]*(m[0]-m[1])*(t[0]+t[1])
    res = (first+np.sqrt(root))/(2*(t[0]+t[1]))
    if res < 0:
        return 0
    return res

@njit #################### needs more testing, can do without error?
def b_maxL_3(m, t, C):
    mt = m[0] + m[1] + m[2]
    tt = t[0] + t[1] + t[2]
    Ct = C[0] + C[1] + C[2]
    a = -tt
    b = -tt*mt + Ct
    c = Ct*mt - C[0]*m[0] - C[1]*m[1] - C[2]*m[2] -tt*(m[0]*m[1] + m[1]*m[2] + m[2]*m[0])
    d = C[0]*m[1]*m[2] + C[1]*m[2]*m[0] + C[2]*m[0]*m[1] - tt*m[0]*m[1]*m[2]
    D0 = b**2 - 3*a*c
    D1 = 2*b**3 - 9*a*b*c + 27*(a**2)*d
        
    if D0 == 0. and D1 == 0.:
        return -b/(3*a)
    
    C0 = ((D1 + np.sqrt(D1**2 - 4*D0**3 + 0j)) / 2)**(1/3)
    
    if C0 == 0:
        C0 = ((D1 - np.sqrt(D1**2 - 4*D0**3 + 0j)) / 2)**(1/3)
        
    x0 = -1/(3*a) * (b + C0 + D0/C0)
    
    if x0.real < 0:
        return 0.
    
    return x0.real


@njit
def logLcore(
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
                elif n_p == 3:
                    b = b_maxL_3(m_b, t_b, C_b)
                else:
                    print()
                    print("b_maxL is not defined")
                    print()
                    return 0.
                for m_i in range(n_p):
                    logL += (counts[p_i][m_i][d_i, e_i]*math.log(t_elapsed[p_i][m_i][d_i]*(m[m_i,e_i]+b))
                            -t_elapsed[p_i][0][d_i]*(m[m_i,e_i]+b))
    return logL

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

def generate_resp_mat(
    rmfs,
    len_dets,
    len_ebs,
    len_emod,
    ra,
    dec,
):
    sds = np.empty(0)
    for d in range(len_dets):
        sd = SPIDRM(rmfs[d], ra, dec)
        sds = np.append(sds, sd.matrix.T)
    return sds.reshape((len_dets, len_emod-1, len_ebs-1))

def calc_mahalanobis_dist(summary, cov, true_vals):
    fit_val = np.array([i[1] for i in summary.values()])
    fit_cov = cov[1]
    rel_distance = []
    
    for i in range(len(true_vals)):
        dif = fit_val - true_vals[i]
        
        rel_distance.append(np.sqrt(
            np.linalg.multi_dot([dif, np.linalg.inv(fit_cov), dif])
        ))
        
    return np.array(rel_distance)

class MultinestClusterFit:
    def __init__(
        self,
        pointings,
        source_model,
        energy_range,
        emod,
        binning_func,
        true_values=None,
        folder=None,
    ):
        self._pointings = pointings
        self._source_model = source_model
        self._binning_func = binning_func
        self._energy_range = energy_range
        self._emod = emod
        
        self._true_values = true_values
        self.set_folder(folder)
        
        self._prepare_fit_data()
        
        self._find_updatable_sources()
        
        self._initialize_resp_mats()
        
        self._run_multinest()
                
        self._extract_parameter_names_simple()
        self._parameter_names.extend(["$z$"])
            
        self._cc = ChainConsumer()
        self._chain = np.loadtxt('./chains/1-post_equal_weights.dat')
        self._cc.add_chain(self._chain, parameters=self._parameter_names, name='fit')
        
    def _run_multinest(self):
        num_sources = len(self._source_model.sources)
        
        def logLba_mult(trial_values, ndim=None, params=None):
            spec_binned = np.zeros((num_sources, len(self._emod)-1))
            for i, parameter in enumerate(self._source_model.free_parameters.values()):
                parameter.value = trial_values[i]
            for i, source in enumerate(self._source_model.sources.values()):
                spec = source(self._emod)
                spec_binned[i,:] = (self._emod[1:]-self._emod[:-1])*(spec[:-1]+spec[1:])/2
            if 1 in self._updatable_sources:
                self._update_resp_mats()
            return logLcore(
                spec_binned,
                self._pointings,
                self._dets,
                self._resp_mats,
                num_sources,
                self._t_elapsed,
                self._counts,
            )
        
        def prior(params, ndim=None, nparams=None):
            for i, parameter in enumerate(self._source_model.free_parameters.values()):
                try:
                    params[i] = parameter.prior.from_unit_cube(params[i])

                except AttributeError:
                    raise RuntimeError(
                        "The prior you are trying to use for parameter %s is "
                        "not compatible with sampling from a unitcube"
                        % parameter.path
                    )

        num_params = len(self._source_model.free_parameters)
        
        # ###
        # trial_values = [4.5e-3, -2.08, 7e-3, -2.5]
        # spec_binned = np.zeros((num_sources, len(self._emod)-1))
        # for i, parameter in enumerate(self._source_model.free_parameters.values()):
        #     parameter.value = trial_values[i]
        # for i, source in enumerate(self._source_model.sources.values()):
        #     spec = source(self._emod)
        #     spec_binned[i,:] = (self._emod[1:]-self._emod[:-1])*(spec[:-1]+spec[1:])/2
        # print(logLcore(
        #     spec_binned,
        #     self._pointings,
        #     self._dets,
        #     self._resp_mats,
        #     num_sources,
        #     self._t_elapsed,
        #     self._counts,
        # ))
        # return None
        # ###

        if not os.path.exists("./chains"):
            os.mkdir("chains")
        sampler = pymultinest.run(
            logLba_mult, prior, num_params, num_params, n_live_points=800, resume=False, verbose=True
        )
    
    def _prepare_fit_data(self):
        ebs = []
        counts = []
        dets = []
        t_elapsed = []
        t_start = []
            
        for combination in self._pointings:
            c_time_start, c_time_elapsed = [], []
            for p_i, pointing in enumerate(combination):
                time_start, time_elapsed, energy_bins, counts_f = extract_pointing_info(pointing[1], pointing[0])
                c_time_start.append(time_start)
                dets_temp = get_live_dets(time=time_start, event_types=["single"])
                c_time_elapsed.append(time_elapsed[dets_temp])
                
                if p_i == 0:
                    dets_0 = dets_temp
                    energy_bins_0 = energy_bins
                    c_counts_f = counts_f[dets_0]
                else:
                    c_counts_f = np.append(c_counts_f, counts_f[dets_0], axis=0)
                    assert np.array_equal(dets_0, dets_temp), f"Active detectors are not the same for {combination[0][0]} and {combination[p_i][0]}"
                    assert np.array_equal(energy_bins_0, energy_bins), f"Energy bins are not the same for {combination[0][0]} and {combination[p_i][0]}"
                
            eb, c = self._binning_func(
                energy_bins_0,
                c_counts_f,
                self._energy_range
            )
            nd = len(dets_0)
            counts.append(tuple([c[i*nd : (i+1)*nd] for i in range(len(combination))]))
            ebs.append(eb)
            
            t_start.append(tuple(c_time_start))
            dets.append(dets_0)
            t_elapsed.append(tuple(c_time_elapsed))
                
        self._ebs = tuple(ebs) 
        self._counts = tuple(counts)
        self._dets = tuple(dets)
        self._t_elapsed = tuple(t_elapsed)
        self._t_start = tuple(t_start)
    
    def _initialize_resp_mats(self):
        # index order: tuple(combination, source, pointing, np_array(dets, e_in, e_out))
        resp_mats = []
        rmfs = []
        
        for count, combination in enumerate(self._pointings):
            source_resp_mats = []
            
            dets = self._dets[count]
            ebs = self._ebs[count]
            
            for source in self._source_model.sources.values():
                combination_resp_mats = []
                combination_rmfs = []
                
                for pointing in range(len(combination)):
                    time = self._t_start[count][pointing]
                    version = find_response_version(time)
                    rsp_base = rsp_bases[version]
                    
                    pointing_rmfs = []
                    for d in dets:
                        pointing_rmfs.append(ResponseRMFGenerator.from_time(time, d, ebs, self._emod, rsp_base))
                    pointing_rmfs = tuple(pointing_rmfs)
                                        
                    combination_resp_mats.append(
                        generate_resp_mat(
                            pointing_rmfs,
                            len(dets),
                            len(ebs),
                            len(self._emod),
                            source.position.get_ra(),
                            source.position.get_dec(),
                        )
                    )
                    combination_rmfs.append(pointing_rmfs)
                    
                source_resp_mats.append(tuple(combination_resp_mats))
                    
            resp_mats.append(tuple(source_resp_mats))
            rmfs.append(tuple(combination_rmfs))
            
        self._resp_mats = tuple(resp_mats)
        if 1 in self._updatable_sources:
            self._updatable_rmfs = tuple(rmfs)

    def _update_resp_mats(self):
        for count, combination in enumerate(self._pointings):
            for source_num, source in enumerate(self._source_model.sources.values()):
                if self._updatable_sources[source_num] == 1:
                    for pointing in range(len(combination)):
                        self._resp_mats[count][source_num][pointing][:,:,:] = generate_resp_mat(
                            self._updatable_rmfs[count][pointing],
                            len(self._dets[count]),
                            len(self._ebs[count]),
                            len(self._emod),
                            source.position.get_ra(),
                            source.position.get_dec(),
                        )

    def _find_updatable_sources(self):
        keywords = ["position"]
        self._updatable_sources = np.zeros(len(self._source_model.sources), np.int8)
        for s_i, source in enumerate(self._source_model.sources.values()):
            for parameter in source.free_parameters.values():
                first_pos = parameter.path.find(".")
                second_pos = parameter.path.find(".", first_pos+1)
                if parameter.path[first_pos+1 : second_pos] in keywords:
                    self._updatable_sources[s_i] = 1

    def parameter_fit_distribution(self):
        assert not self._folder is None, "folder is not set"
        
        fig = self._cc.plotter.plot(
            parameters=self._parameter_names[:-1],
            # truth={'Crab K':true_values_main[0,0], 'Crab index':true_values_main[0,1]},
            figsize=1.5
        )
        
        plt.savefig(f"{self._folder}/parameter_fit_distributions.pdf")
        plt.close()
        
    def text_summaries(
        self,
        reference_values=True,
        pointing_combinations=True,
        parameter_fit_constraints=True
    ):
        assert not self._folder is None, "folder is not set"
        
        
        if reference_values:
            assert not self._true_values is None, "true_values not set"
            summary = self._cc.analysis.get_summary(parameters=self._true_values[0])
            cov = self._cc.analysis.get_covariance(parameters=self._true_values[0])
            rel_distances = calc_mahalanobis_dist(summary, cov, self._true_values[1])
            
            with open(f"{self._folder}/reference_values", "w") as f:
                f.write(f"{' : '.join(self._true_values[0])} : Rel. Dist.\n")
                for i in range(self._true_values[1].shape[0]):
                    f.write(f"{' : '.join([f'{j:.3}' for j in self._true_values[1][i,:]])} : {rel_distances[i]:.3}\n")
                
        if pointing_combinations:
            with open(f"{self._folder}/pointing_combinations", "w") as f:
                for combination in self._pointings:
                    f.write(f'{"  ".join(i[0] for i in combination)}\n')
        
        if parameter_fit_constraints:
            summary = self._cc.analysis.get_summary(parameters=self._parameter_names[:-1])
            with open(f"{self._folder}/parameter_fit_constraints", "w") as f:
                for param in self._parameter_names[:-1]:
                    f.write(f"{param}:\n")
                    try:
                        f.write(f"{summary[param][0]:.5}  {summary[param][1]:.5}  {summary[param][2]:.5}\n")
                    except:
                        f.write(f"None  {summary[param][1]:.5}  None\n")
    
    def ppc(
        self,
        count_energy_plots=True,
        qq_plots=True
    ):
        assert self._folder is not None, "folder is not set"
        
        s, b = self._calc_rates()
        
        for c_i, combination in enumerate(self._pointings):
            for p_i in range(len(combination)):
            
                if count_energy_plots:
                    self._count_energy_plot(
                        b[c_i][p_i],
                        s[c_i][p_i],
                        self._ebs[c_i],
                        self._counts[c_i][p_i],
                        self._dets[c_i],
                        combination[p_i][0]
                    )
                if qq_plots:
                    self._qq_plot(
                        b[c_i][p_i],
                        s[c_i][p_i],
                        self._counts[c_i][p_i],
                        self._dets[c_i],
                        combination[p_i][0]
                    )
            
    def _calc_rates(self):
        source_rate = []
        background_rate = []
        for c_i, combination in enumerate(self._pointings):
            source_rate.append(np.zeros((len(combination), len(self._dets[c_i]), len(self._ebs[c_i])-1, len(self._chain))))
            background_rate.append(np.zeros((len(self._dets[c_i]), len(self._ebs[c_i])-1, len(self._chain))))
        
        num_sources = len(self._source_model.sources)
        
        for p_i, params in enumerate(self._chain):
            spec_binned = np.zeros((num_sources, len(self._emod)-1))
            for fp_i, parameter in enumerate(self._source_model.free_parameters.values()):
                parameter.value = params[fp_i]
            for s_i, source in enumerate(self._source_model.sources.values()):
                spec = source(self._emod)
                spec_binned[s_i,:] = (self._emod[1:]-self._emod[:-1])*(spec[:-1]+spec[1:])/2
            if 1 in self._updatable_sources:
                self._update_resp_mats()
            
            for c_i, combination in enumerate(self._pointings):
                for d_i in range(len(self._dets[c_i])):
                    for s_i in range(num_sources):
                        for m_i in range(len(combination)):
                            source_rate[c_i][m_i,d_i,:,p_i] += np.dot(spec_binned[s_i,:], self._resp_mats[c_i][s_i][m_i][d_i])
                    for e_i in range(len(self._ebs[c_i])-1):
                        s_b = np.array([source_rate[c_i][i,d_i,e_i,p_i] for i in range(len(combination))])
                        t_b = np.array([self._t_elapsed[c_i][i][d_i] for i in range(len(combination))])
                        C_b = np.array([self._counts[c_i][i][d_i, e_i] for i in range(len(combination))])
                        if len(combination) == 2:
                            background_rate[c_i][d_i,e_i,p_i] = b_maxL_2(s_b, t_b, C_b)
                        elif len(combination) == 3:
                            background_rate[c_i][d_i,e_i,p_i] = b_maxL_3(s_b, t_b, C_b)
                        
        source_rates = []
        background_rates = []
        for c_i, combination in enumerate(self._pointings):
            c_source_rates = []
            c_background_rates = []
            
            for p_i in range(len(combination)):
                c_source_rates.append(
                    np.average(source_rate[c_i][p_i], axis=2) * self._t_elapsed[c_i][p_i][:,np.newaxis]
                )
                c_background_rates.append(
                    np.average(background_rate[c_i], axis=2) * self._t_elapsed[c_i][p_i][:,np.newaxis]
                )
            
            source_rates.append(c_source_rates)
            background_rates.append(c_background_rates)

        return source_rates, background_rates

    def _count_energy_plot(
        self,
        b,
        s,
        eb,
        c,
        dets,
        name
    ):
        fig, axes = plt.subplots(5,4, sharex=True, sharey=True, figsize=(10,10))
        axes = axes.flatten()
        
        predicted = b + s
        predicted_lower = poisson.ppf(0.16, predicted)
        predicted_upper = poisson.ppf(0.84, predicted)
        counts = c
        
        i=0
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
        
        fig.savefig(f"{self._folder}/{name}_count_energy.pdf")
        plt.close()
    
    def _qq_plot(
        self,
        b,
        s,
        c,
        dets,
        name
    ):
        fig, axes = plt.subplots(5,4, sharex=True, sharey=True, figsize=(10,10))
        axes = axes.flatten()
        
        p = b + s
        predicted = np.cumsum(p, axis=1)
        predicted_lower = np.cumsum(poisson.ppf(0.16, p), axis=1)
        predicted_upper = np.cumsum(poisson.ppf(0.84, p), axis=1)
        counts = np.cumsum(c, axis=1)
        ma = np.amax(
            np.array([np.amax(counts, axis=1), np.amax(predicted, axis=1)]),
            axis=0
        )
        
        i=0
        for d in range(19):
            axes[d].text(.5,.9,f"Det {d}",horizontalalignment='center',transform=axes[d].transAxes)
            if d in dets:
                line2, = axes[d].plot([0, ma[i]], [0, ma[i]], ls="--", c="k")
                line1, = axes[d].plot(counts[i], predicted[i], c="r")
                axes[d].fill_between(counts[i], predicted_lower[i], predicted_upper[i], color="r", alpha=0.5)
                i += 1
        plt.subplots_adjust(hspace=0, wspace=0)
        plt.subplots_adjust(hspace=0, top=0.96, bottom=0.1)
                
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Cumulative Real Counts")
        plt.ylabel("Cumulative Predicted Counts", labelpad=27)
        
        fig.savefig(f"{self._folder}/{name}_qq.pdf")
        plt.close()
        
    def _extract_parameter_names_simple(self):
        self._parameter_names = []
        for full_name in self._source_model.free_parameters.keys():
            source = full_name[ : full_name.find(".")]
            source = source[1:] if source[0]=="_" else source
            source = source.replace("__", "+").replace("_", " ")
            parameter = full_name[-1 * full_name[::-1].find(".") : ]
            self._parameter_names.extend([f"{source} {parameter}"])
    
    def set_folder(self, folder):
        if not folder is None:
            if not os.path.exists(f"./{folder}"):
                os.mkdir(folder)
        self._folder = folder
