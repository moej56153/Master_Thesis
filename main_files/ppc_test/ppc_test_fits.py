import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle

def fit_1662_correct():
    data_folder = "./main_files/SPI_data/1662"
    
    folder = "./main_files/ppc_test/ppc_test_1662_correct"

    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.5,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=600.,
        cluster_size_range=(2,2),
    ).pointings
    save_clusters(pointings, folder)

    pointings = load_clusters(folder)
    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def fit_1664_correct():
    data_folder = "./main_files/SPI_data/1664"
    
    folder = "./main_files/ppc_test/ppc_test_1664_correct"

    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.5,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=600.,
        cluster_size_range=(2,2),
    ).pointings
    save_clusters(pointings, folder)

    pointings = load_clusters(folder)
    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def fit_1667_correct():
    data_folder = "./main_files/SPI_data/1667"
    
    folder = "./main_files/ppc_test/ppc_test_1667_correct"

    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.5,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=600.,
        cluster_size_range=(2,2),
    ).pointings
    save_clusters(pointings, folder)

    pointings = load_clusters(folder)
    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def fit_1662_simple():
    folder = "./main_files/ppc_test/ppc_test_1662_simple"

    folder2 = "./main_files/ppc_test/ppc_test_1662_correct"
    pointings = load_clusters(folder2)

    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def fit_1664_simple():
    folder = "./main_files/ppc_test/ppc_test_1664_simple"

    folder2 = "./main_files/ppc_test/ppc_test_1664_correct"
    pointings = load_clusters(folder2)

    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def fit_1667_simple():
    folder = "./main_files/ppc_test/ppc_test_1667_simple"

    folder2 = "./main_files/ppc_test/ppc_test_1667_correct"
    pointings = load_clusters(folder2)

    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)
        
def fit_1662_individual():
    big_folder = "./main_files/ppc_test/ppc_test_1662_individual"

    if not os.path.exists(f"./{big_folder}"):
        os.mkdir(big_folder)

    folder2 = "./main_files/ppc_test/ppc_test_1662_correct"

    pointings2 = load_clusters(folder2)
    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    for i in range(len(pointings2)):
        pointings = (pointings2[i],)
        folder = f"{big_folder}/{pointings[0][0][0]}_{pointings[0][1][0]}"
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 81.5),
            np.geomspace(18, 150, 50),
            log_binning_function_for_x_number_of_bins(125),
            folder=folder,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()

        p = ["Crab K", "Crab index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{folder}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)
        
def fit_1664_individual():
    big_folder = "./main_files/ppc_test/ppc_test_1664_individual"

    if not os.path.exists(f"./{big_folder}"):
        os.mkdir(big_folder)

    folder2 = "./main_files/ppc_test/ppc_test_1664_correct"

    pointings2 = load_clusters(folder2)
    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    for i in range(len(pointings2)):
        pointings = (pointings2[i],)
        folder = f"{big_folder}/{pointings[0][0][0]}_{pointings[0][1][0]}"
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 81.5),
            np.geomspace(18, 150, 50),
            log_binning_function_for_x_number_of_bins(125),
            folder=folder,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()

        p = ["Crab K", "Crab index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{folder}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)
            
def fit_1667_individual():
    big_folder = "./main_files/ppc_test/ppc_test_1667_individual"

    if not os.path.exists(f"./{big_folder}"):
        os.mkdir(big_folder)

    folder2 = "./main_files/ppc_test/ppc_test_1667_correct"

    pointings2 = load_clusters(folder2)
    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    for i in range(len(pointings2)):
        pointings = (pointings2[i],)
        folder = f"{big_folder}/{pointings[0][0][0]}_{pointings[0][1][0]}"
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 81.5),
            np.geomspace(18, 150, 50),
            log_binning_function_for_x_number_of_bins(125),
            folder=folder,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()

        p = ["Crab K", "Crab index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{folder}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)   

def fit_1667_simple_wo_outliers():
    folder = "./main_files/ppc_test/ppc_test_1667_simple_wo_outliers"

    folder2 = "./main_files/ppc_test/ppc_test_1667_correct"
    pointings2 = load_clusters(folder2)

    pointings = []

    bad_pointings = (
        "166700470010",
        "166700520010",
        "166700480010",
        "166700530010",
        "166700490010",
        "166700550010",
        "166700500010",
        "166700560010",
        "166700620010",
        "166700640010",
        "166700730010",
        "166700750010",
    )

    for cluster in pointings2:
        if cluster[0][0] in bad_pointings:
            continue
        else:
            pointings.append(cluster)
            
    pointings = tuple(pointings)

    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)

def fit_1667_simple_triple():
    data_folder = "./main_files/SPI_data/1667"
    
    folder = "./main_files/ppc_test/ppc_test_1667_simple_triple"

    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.0,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=600.,
        cluster_size_range=(3,3),
    ).pointings
    save_clusters(pointings, folder)

    pointings = load_clusters(folder)

    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (20., 81.5),
        np.geomspace(18, 150, 50),
        log_binning_function_for_x_number_of_bins(125),
        folder=folder,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()

    p = ["Crab K", "Crab index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{folder}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)

def fit_1667_simple_triple_individual():
    big_folder = "./main_files/ppc_test/ppc_test_1667_triple_individual"

    if not os.path.exists(f"./{big_folder}"):
        os.mkdir(big_folder)

    folder2 = "./main_files/ppc_test/ppc_test_1667_simple_triple"

    pointings2 = load_clusters(folder2)
    source_model = define_sources((
        (crab_pl_fixed_pos, (40,)),
    ))

    for i in range(len(pointings2)):
        pointings = (pointings2[i],)
        folder = f"{big_folder}/{pointings[0][0][0]}_{pointings[0][1][0]}_{pointings[0][2][0]}"
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (20., 81.5),
            np.geomspace(18, 150, 50),
            log_binning_function_for_x_number_of_bins(125),
            folder=folder,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()

        p = ["Crab K", "Crab index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{folder}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)   

fit_1667_simple_triple_individual()