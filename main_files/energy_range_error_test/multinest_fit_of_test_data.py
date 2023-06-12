import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle

data_folder = "./main_files/energy_range_error_test/bkg_20_lin_source_8__6_data"

    
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

pointings = ((('037400020010', './main_files/energy_range_error_test/bkg_20_lin_source_8__6_data'),
  ('037400030010', './main_files/energy_range_error_test/bkg_20_lin_source_8__6_data')),)

source_model = define_sources((
    (simulated_linear_0374, ()),
))

d_path = "./main_files/energy_range_error_test/bkg_20_lin_source_8__6_fit"

import os
if not os.path.exists(f"./{d_path}"):
    os.mkdir(f"./{d_path}")

energy_ranges = [
    (18., 2000.),
    (50., 2000.),
    (18., 1000.),
]


for i in range(len(energy_ranges)):
        
    temp_path = f"{d_path}/e{int(energy_ranges[i][0])}_{int(energy_ranges[i][1])}"

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        energy_ranges[i],
        np.geomspace(18, 3000, 50),
        log_binning_function_for_x_number_of_bins(125),
        # true_values=true_values(),
        folder=temp_path,
        source_spectrum_powerlaw_binning=False
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    # multinest_fit.ppc()

    # print(multinest_fit._cc._all_parameters)

    # p = ["Simulated Source 0374 b"]
    # val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    # cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    # with open(f"./{temp_path}/crab_parameters.pickle", "wb") as f:
    #     pickle.dump((val, cov), f)