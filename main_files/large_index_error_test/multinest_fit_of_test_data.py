import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle

data_folder = "./main_files/energy_range_error_test/bkg_20000_lin_source_8__6_data"

    
# pointings = PointingClusters(
#     (data_folder,),
#     min_angle_dif=1.5,
#     max_angle_dif=7.5,
#     max_time_dif=0.2,
#     radius_around_source=10.,
#     min_time_elapsed=300.,
#     cluster_size_range=(2,2),
#     center_ra=10.,
#     center_dec=-40.,
# ).pointings
# save_clusters(pointings, data_folder)

# pointings = load_clusters(data_folder)

pointings = ((('037400020010', "./main_files/large_index_error_test/bkg_20000_pl_source_i_8_K_8__4"),
  ('037400030010', "./main_files/large_index_error_test/bkg_20000_pl_source_i_8_K_8__4")),)

source_model = define_sources((
    (simulated_pl_0374, (100.,)),
))

d_path = "./main_files/large_index_error_test/bkg_20000_pl_source_i_8_K_8__4"

import os
if not os.path.exists(f"./{d_path}"):
    os.mkdir(f"./{d_path}")



        
temp_path = d_path

multinest_fit = MultinestClusterFit(
    pointings,
    source_model,
    (None, None),
    np.geomspace(18, 3000, 50),
    log_binning_function_for_x_number_of_bins(25),
    # true_values=true_values(),
    folder=temp_path,
    source_spectrum_powerlaw_binning=True
)

multinest_fit.parameter_fit_distribution()
multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
multinest_fit.ppc()

# print(multinest_fit._cc._all_parameters)

# p = ["Simulated Source 0374 b"]
# val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
# cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

# with open(f"./{temp_path}/crab_parameters.pickle", "wb") as f:
#     pickle.dump((val, cov), f)


with open(f"{d_path}/resp_mats.pickle", "wb") as f:
    pickle.dump((
            multinest_fit._pointings,
            multinest_fit._dets,
            multinest_fit._resp_mats,
            len(multinest_fit._source_model.sources),
            multinest_fit._t_elapsed,
            multinest_fit._counts,
        ), f)