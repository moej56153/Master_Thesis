import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *



# ra, dec = 155., 75.

def pyspi_real_bkg_fit_0374_pre_ppc():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/pre_ppc/{rev}"
    
    if not os.path.exists(f"{data_path}"):
        os.mkdir(f"{data_path}")
    
    # pointings = PointingClusters(
    #         (data_path,),
    #         min_angle_dif=2.5,
    #         max_angle_dif=20.,
    #         max_time_dif=0.2,
    #         radius_around_source=10.,
    #         min_time_elapsed=300.,
    #         cluster_size_range=(2,2),
    #         center_ra=ra,
    #         center_dec=dec,
    #     ).pointings
    # save_clusters(pointings, data_path)
    
    pointings = load_clusters(data_path)
    
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
        folder=data_path,
    )
    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()
    
pyspi_real_bkg_fit_0374_pre_ppc()











