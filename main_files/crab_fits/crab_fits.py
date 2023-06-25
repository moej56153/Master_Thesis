import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle

data_folder = "./main_files/SPI_data"

revolutions = ["0043","0044","0045","0422","0966","0967","0970","1327","1328","1657","1658","1661","1662","1664","1667","1996","1999","2000"]

revolutions += ["1268","1269","1278","1461","1462","1466","1468","1515","1516","1520","1528","1593","1597","1598","1599","1781","1784","1785","1789","1850","1856","1857","1921","1925","1927","1930","2058","2062","2063","2066"]


fit_folder = "./main_files/crab_fits/general_tests"

if not os.path.exists(f"{fit_folder}"):
    os.mkdir(f"{fit_folder}")


######################### check to make sure fit functions have everything!!!!!!!!!!!!!!!!!!!!

def crab_lower_band_ind_rev_fits(data_folder=data_folder, revolutions=revolutions, fit_folder=fit_folder):
    source_model = define_sources((
        (crab_lower_band, (100,)),
    ))
    
    
    for rev in revolutions:
        path_d = f"{data_folder}/{rev}"
        path_f = f"{fit_folder}/{rev}_lower_band"
        if not os.path.exists(f"{path_f}"):
            os.mkdir(f"{path_f}")
        
        pointings = load_clusters(path_d)
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 400.),
            np.geomspace(18, 600, 50),
            log_binning_function_for_x_number_of_bins(70),
            # true_values=true_values(),
            folder=path_f,
        )
        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        multinest_fit.ppc()
        
def crab_pl_ind_rev_fits(data_folder=data_folder, revolutions=revolutions, fit_folder=fit_folder):
    source_model = define_sources((
        (crab_pl_fixed_pos, (100,)),
    ))
    
    
    for rev in revolutions:
        path_d = f"{data_folder}/{rev}"
        path_f = f"{fit_folder}/{rev}_pl"
        if not os.path.exists(f"{path_f}"):
            os.mkdir(f"{path_f}")
        
        pointings = load_clusters(path_d)
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 400.),
            np.geomspace(18, 600, 50),
            log_binning_function_for_x_number_of_bins(70),
            # true_values=true_values(),
            folder=path_f,
        )
        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        multinest_fit.ppc()
        
def crab_sm_br_pl_ind_rev_fits(data_folder=data_folder, revolutions=revolutions, fit_folder=fit_folder):
    source_model = define_sources((
        (crab_sm_br_pl, (100,)),
    ))
    
    
    for rev in revolutions:
        path_d = f"{data_folder}/{rev}"
        path_f = f"{fit_folder}/{rev}_sm_br_pl"
        if not os.path.exists(f"{path_f}"):
            os.mkdir(f"{path_f}")
        
        pointings = load_clusters(path_d)
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 400.),
            np.geomspace(18, 600, 50),
            log_binning_function_for_x_number_of_bins(70),
            # true_values=true_values(),
            folder=path_f,
        )
        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        multinest_fit.ppc()
        
def crab_pulsar_sm_br_pl_ind_rev_fits(data_folder=data_folder, revolutions=revolutions, fit_folder=fit_folder):
    source_model = define_sources((
        (crab_sm_br_pl, (100,)),
        (s_1A_0535_262_pl, (100,)),
    ))
    
    
    for rev in revolutions:
        path_d = f"{data_folder}/{rev}"
        path_f = f"{fit_folder}/{rev}_sm_br_pl_w_p"
        if not os.path.exists(f"{path_f}"):
            os.mkdir(f"{path_f}")
        
        pointings = load_clusters(path_d)
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 400.),
            np.geomspace(18, 600, 50),
            log_binning_function_for_x_number_of_bins(70),
            # true_values=true_values(),
            folder=path_f,
        )
        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        multinest_fit.ppc()

def crab_pulsar_pl_ind_rev_fits(data_folder=data_folder, revolutions=revolutions, fit_folder=fit_folder):
    source_model = define_sources((
        (crab_pl_fixed_pos, (100,)),
        (s_1A_0535_262_pl, (100,)),
    ))
    
    
    for rev in revolutions:
        path_d = f"{data_folder}/{rev}"
        path_f = f"{fit_folder}/{rev}_pl_w_p"
        if not os.path.exists(f"{path_f}"):
            os.mkdir(f"{path_f}")
        
        pointings = load_clusters(path_d)
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (30., 81.5),
            np.geomspace(25, 150, 35),
            log_binning_function_for_x_number_of_bins(70),
            # true_values=true_values(),
            folder=path_f,
        )
        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        multinest_fit.ppc()
        
        p = ["Crab K", "Crab index", "A 0535 262 K", "A 0535 262 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{path_f}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)


crab_pulsar_pl_ind_rev_fits()


def crab_pointing_clustering(data_folder=data_folder, revolutions=revolutions):
    bad_revs = []
    for rev in revolutions:
        try:
            path_d = f"{data_folder}/{rev}"
            
            pointings = PointingClusters(
                (path_d,),
                min_angle_dif=2.5,
                max_angle_dif=10.,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
                center_ra=83.6333,
                center_dec=22.0144,
            ).pointings
            save_clusters(pointings, path_d)
        except:
            bad_revs.append(rev)
            
    with open(f"./main_files/crab_fits/bad_revs.txt", "w") as f:
        for rev in bad_revs:
            f.write(f"{rev}\n")

# crab_pointing_clustering()  