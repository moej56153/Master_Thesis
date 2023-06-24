import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from MultinestClusterFit import *
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle
import matplotlib.pyplot as plt

l_path = "./main_files/ppc_test_new"

def pyspi_const_bkg_fit_0374_pre_ppc():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_const_bkg/{rev}"
    
    fit_path = f"{l_path}/const_bkg"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    pointings = PointingClusters(
            (data_path,),
            min_angle_dif=1.5,
            max_angle_dif=10.,
            max_time_dif=0.2,
            radius_around_source=10.,
            min_time_elapsed=300.,
            cluster_size_range=(2,2),
            center_ra=ra,
            center_dec=dec,
        ).pointings
    save_clusters(pointings, fit_path)
    
    pointings = load_clusters(fit_path)
    
    if rev=="0374":
        s = simulated_pl_0374
    elif rev=="1380":
        s = simulated_pl_1380
    
    source_model = define_sources((
        (s, (40,)),
    ))
    
    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (30., 400.,),
        np.geomspace(18, 600, 50),
        log_binning_function_for_x_number_of_bins(70),
        # true_values=true_values(),
        folder=fit_path,
    )
    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()
    
    p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{fit_path}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
    with open(f"{fit_path}/temp_dump","wb") as f:
        pickle.dump(
            (
                multinest_fit._pointings,
                multinest_fit._source_model,
                multinest_fit._energy_range,
                multinest_fit._emod,
                multinest_fit._folder,
                multinest_fit._cc,
                multinest_fit._chain,
                multinest_fit._parameter_names,
                multinest_fit._dets,
                multinest_fit._ebs,
                multinest_fit._t_elapsed,
                multinest_fit._resp_mats,
                multinest_fit._updatable_sources,
                multinest_fit._counts
            ),
            f
        )


def calc_exp_rates():
    max_posterior_samples = 300
    fit_path = f"{l_path}/const_bkg"
    with open(f"{fit_path}/temp_dump", "rb") as f:
        (
            pointings,
            source_model,
            energy_range,
            emod,
            folder,
            cc,
            chain,
            parameter_names,
            dets,
            ebs,
            t_elapsed,
            resp_mats,
            updatable_sources,
            counts
        ) = pickle.load(f)
        
    source_rate = []
    background_rate = []
    
    b_range = [None, None]
    s_range = [None, None]
    t_range = [None, None]
    
    if len(chain) < max_posterior_samples:
        print(f"Using all {len(chain)} equal-weight posterior values.")
        max_posterior_samples = len(chain)
    posterior_samples = chain[
        np.random.choice(len(chain), max_posterior_samples, replace=False)
    ]
    
    for c_i, combination in enumerate(pointings):
        source_rate.append(np.zeros((len(combination), len(dets[c_i]), len(ebs[c_i])-1, len(posterior_samples))))
        background_rate.append(np.zeros((len(dets[c_i]), len(ebs[c_i])-1, len(posterior_samples))))
    
    num_sources = len(source_model.sources)
    
    for p_i, params in enumerate(posterior_samples):
        spec_binned = np.zeros((num_sources, len(emod)-1))
        for fp_i, parameter in enumerate(source_model.free_parameters.values()):
            parameter.value = params[fp_i]
        for s_i, source in enumerate(source_model.sources.values()):
            spec = source(emod)
            # if source_spectrum_powerlaw_binning:
            spec_binned[s_i,:] = powerlaw_binned_spectrum(emod, spec)
            # else:
                # spec_binned[s_i,:] = (emod[1:]-emod[:-1])*(spec[:-1]+spec[1:])/2
        # if 1 in updatable_sources:
        #     update_resp_mats()
        
        for c_i, combination in enumerate(pointings):
            for d_i in range(len(dets[c_i])):
                for s_i in range(num_sources):
                    for m_i in range(len(combination)):
                        source_rate[c_i][m_i,d_i,:,p_i] += np.dot(spec_binned[s_i,:], resp_mats[c_i][s_i][m_i][d_i])
                        
                        if t_range[0] is None:
                            t_range = [np.amin(t_elapsed[c_i][m_i][d_i]), np.amax(t_elapsed[c_i][m_i][d_i])]
                        else:
                            min = np.amin(t_elapsed[c_i][m_i][d_i])
                            max = np.amax(t_elapsed[c_i][m_i][d_i])
                            if min < t_range[0]:
                                t_range[0] = min
                            elif max > t_range[1]:
                                t_range[1] = max
                        
                for e_i in range(len(ebs[c_i])-1):
                    s_b = np.array([source_rate[c_i][i,d_i,e_i,p_i] for i in range(len(combination))])
                    t_b = np.array([t_elapsed[c_i][i][d_i] for i in range(len(combination))])
                    C_b = np.array([counts[c_i][i][d_i, e_i] for i in range(len(combination))])
                    if len(combination) == 2:
                        background_rate[c_i][d_i,e_i,p_i] = b_maxL_2(s_b, t_b, C_b)
                    # elif len(combination) == 3:
                    #     background_rate[c_i][d_i,e_i,p_i] = b_maxL_3(s_b, t_b, C_b)
                        
                    if b_range[0] is None:
                        b_range = [background_rate[c_i][d_i,e_i,p_i], background_rate[c_i][d_i,e_i,p_i]]
                    else:
                        if background_rate[c_i][d_i,e_i,p_i] < b_range[0]:
                            b_range[0] = background_rate[c_i][d_i,e_i,p_i]
                        elif background_rate[c_i][d_i,e_i,p_i] > b_range[1]:
                            b_range[1] = background_rate[c_i][d_i,e_i,p_i]
                            
            if s_range[0] is None:
                s_range = [np.amin(source_rate[c_i][:,:,:,p_i]), np.amax(source_rate[c_i][:,:,:,p_i])]
            else:
                min = np.amin(source_rate[c_i][:,:,:,p_i])
                max = np.amax(source_rate[c_i][:,:,:,p_i])
                if min < s_range[0]:
                    s_range[0] = min
                elif max > s_range[1]:
                    s_range[1] = max
            
    ### correct ppc         
    background_rate = tuple(background_rate)
    source_rate = tuple(source_rate)
    
    with open(f"{fit_path}/temp_dump2","wb") as f:
        pickle.dump(
            (
                background_rate,
                source_rate,
                b_range,
                s_range,
                t_range,
                posterior_samples
            ),
            f
        )
   
# calc_exp_rates()

test_c = 1
# test_i = 1
test_d = 0
test_e = 50
    
def calc_var_mat():
    fit_path = f"{l_path}/const_bkg"
    
    with open(f"{fit_path}/temp_dump", "rb") as f:
        (
            pointings,
            source_model,
            energy_range,
            emod,
            folder,
            cc,
            chain,
            parameter_names,
            dets,
            ebs,
            t_elapsed,
            resp_mats,
            updatable_sources,
            counts
        ) = pickle.load(f)
        
    with open(f"{fit_path}/temp_dump2", "rb") as f:
        (
            background_rate,
            source_rate,
            b_range,
            s_range,
            t_range,
            posterior_samples
        ) = pickle.load(f)
        
    
    
    b_int_funcs = (interpolate_linear, interpolate_linear, interpolate_logarithmic, interpolate_linear, interpolate_logarithmic)
    c_int_funcs = (interpolate_logarithmic, interpolate_linear, interpolate_powerlaw, interpolate_linear, interpolate_logarithmic)
    s_int_funcs = (interpolate_constant, interpolate_linear, interpolate_powerlaw, interpolate_constant, interpolate_constant)
    
    b_num = 7
    s_num = 9
    t_num = 5
    
    input_b = np.geomspace(b_range[0]*0.999, b_range[1]*1.001, b_num)
    if s_range[0] == 0.0:
        input_s = np.geomspace(s_range[1]*0.005, s_range[1]*1.001, s_num-1)
        input_s = np.insert(input_s, 0, 0.0)
    else:
        input_s = np.geomspace(s_range[0]*0.999, s_range[1]*1.001, s_num)
    input_t = np.geomspace(t_range[0]*0.999, t_range[1]*1.001, t_num)
    
    
    dimension_values = (input_b, input_s, input_t, input_s, input_t)
    print("Matrix - B:")
    print(input_b)
    print("Matrix - S:")
    print(input_s)
    print("Matrix - T:")
    print(input_t)
    
    print("Generating Variance Matrix")
    
    variance_matrix = calc_bmaxL_variance_matrix(input_b, input_s, input_t, input_s, input_t)
    
    with open(f"{fit_path}/temp_dump3","wb") as f:
        pickle.dump(
            (
                b_int_funcs,
                c_int_funcs,
                s_int_funcs,
                dimension_values,
                variance_matrix
            ),
            f
        )
        
    

def sample_plot():
    fit_path = f"{l_path}/const_bkg"
    with open(f"{fit_path}/temp_dump", "rb") as f:
        (
            pointings,
            source_model,
            energy_range,
            emod,
            folder,
            cc,
            chain,
            parameter_names,
            dets,
            ebs,
            t_elapsed,
            resp_mats,
            updatable_sources,
            counts
        ) = pickle.load(f)
        
    with open(f"{fit_path}/temp_dump2", "rb") as f:
        (
            background_rate,
            source_rate,
            b_range,
            s_range,
            t_range,
            posterior_samples
        ) = pickle.load(f)
        
    with open(f"{fit_path}/temp_dump3", "rb") as f:
        (
            b_int_funcs,
            c_int_funcs,
            s_int_funcs,
            dimension_values,
            variance_matrix,
        ) = pickle.load(f)
        
    expected_counts = []
    
    b_int_funcs = (interpolate_linear, interpolate_linear, interpolate_logarithmic, interpolate_linear, interpolate_logarithmic)
    c_int_funcs = (interpolate_logarithmic, interpolate_linear, interpolate_powerlaw, interpolate_linear, interpolate_logarithmic)
    s_int_funcs = (interpolate_constant, interpolate_linear, interpolate_powerlaw, interpolate_constant, interpolate_constant)
        
    print("Sampling Count Rates")
    # print(variance_matrix)
    
    for c_i, combination in enumerate(pointings):   
        expected_counts.append(sample_count_rates(
            c_i,
            source_rate,
            background_rate,
            posterior_samples,
            dets,
            ebs,
            t_elapsed,
            variance_matrix,
            dimension_values,
            b_int_funcs,
            c_int_funcs,
            s_int_funcs
        ))
        
    backs0 = background_rate[test_c][test_d,test_e,:] * t_elapsed[test_c][0][test_d]
    sourcs0 = source_rate[test_c][0,test_d,test_e,:] * t_elapsed[test_c][0][test_d]
    counts0 = counts[test_c][0][test_d, test_e]
    
    tots0 = backs0 + sourcs0
    
    plt.hist(tots0, bins=30)
    plt.axvline(counts0)
    plt.savefig(f"{fit_path}/nr_0.pdf")
    
    plt.clf()
    
    backs1 = background_rate[test_c][test_d,test_e,:] * t_elapsed[test_c][1][test_d]
    sourcs1 = source_rate[test_c][1,test_d,test_e,:] * t_elapsed[test_c][1][test_d]
    counts1 = counts[test_c][1][test_d, test_e]
    
    tots1 = backs1 + sourcs1
    
    plt.hist(tots1, bins=30)
    plt.axvline(counts1)
    plt.savefig(f"{fit_path}/nr_1.pdf")
    
    plt.clf()
    
    totsr0 = expected_counts[test_c][0,test_d,test_e,:]
    plt.hist(totsr0, bins=30)
    plt.axvline(counts0)
    plt.savefig(f"{fit_path}/r_0.pdf")
    
    plt.clf()
    
    totsr1 = expected_counts[test_c][1,test_d,test_e,:]
    plt.hist(totsr1, bins=30)
    plt.axvline(counts1)
    plt.savefig(f"{fit_path}/r_1.pdf")



sample_plot()






