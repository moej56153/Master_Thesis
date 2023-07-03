import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle


def fit_identical():
    data_folder = "./main_files/pure_simulation_tests/identical_repeats"

    repeats = 8

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices = pickle.load(f)
        
    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.5,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=300.,
        cluster_size_range=(2,2),
        center_ra=10.,
        center_dec=-40.,
    ).pointings
    save_clusters(pointings, data_folder)

    pointings = load_clusters(data_folder)
    source_model = define_sources((
        (simulated_pl_0374, (200,)),
    ))

    for i in range(repeats):
            
        temp_path = f"{data_folder}/{i}"

        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (None, None),
            np.geomspace(18, 3000, 50),
            log_binning_function_for_x_number_of_bins(125),
            # true_values=true_values(),
            folder=temp_path,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        multinest_fit.ppc()
        
        # print(multinest_fit._cc._all_parameters)
        
        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)

def fit_different_sources():
    data_folder = "./main_files/pure_simulation_tests/different_sources"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices = pickle.load(f)
        

    for i in range(len(source_Ks)):
        for j in range(len(source_indices)):
            
            temp_path = f"{data_folder}/{i}_{j}"

            pointings = PointingClusters(
                (temp_path,),
                min_angle_dif=1.5,
                max_angle_dif=7.5,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
                center_ra=10.,
                center_dec=-40.,
            ).pointings
            save_clusters(pointings, temp_path)

            pointings = load_clusters(temp_path)
            source_model = define_sources((
                (simulated_pl_0374, (100,)),
            ))

            multinest_fit = MultinestClusterFit(
                pointings,
                source_model,
                (None, None),
                np.geomspace(18, 3000, 50),
                log_binning_function_for_x_number_of_bins(125),
                # true_values=true_values(),
                folder=temp_path,
            )

            multinest_fit.parameter_fit_distribution()
            multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
            multinest_fit.ppc()
            
            # print(multinest_fit._cc._all_parameters)
            
            p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
            val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
            cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

            with open(f"./{temp_path}/source_parameters.pickle", "wb") as f:
                pickle.dump((val, cov), f)
                
def fit_second_source_i_1():
    data_folder = "./main_files/pure_simulation_tests/second_source_i_1"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, s_source_decs, s_source_Ks, s_source_index = pickle.load(f)
        

    for i in range(len(s_source_decs)):
        for j in range(len(s_source_Ks)):
            
            temp_path = f"{data_folder}/{i}_{j}"

            pointings = PointingClusters(
                (temp_path,),
                min_angle_dif=1.5,
                max_angle_dif=7.5,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
                center_ra=10.,
                center_dec=-40.,
            ).pointings
            save_clusters(pointings, temp_path)

            pointings = load_clusters(temp_path)
            source_model = define_sources((
                (simulated_pl_0374, (200,)),
            ))

            multinest_fit = MultinestClusterFit(
                pointings,
                source_model,
                (None, None),
                np.geomspace(18, 3000, 50),
                log_binning_function_for_x_number_of_bins(125),
                # true_values=true_values(),
                folder=temp_path,
            )

            multinest_fit.parameter_fit_distribution()
            multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
            # multinest_fit.ppc()
            
            # print(multinest_fit._cc._all_parameters)
            
            p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
            val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
            cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

            with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
                pickle.dump((val, cov), f)
                
def fit_second_source_i_2():
    data_folder = "./main_files/pure_simulation_tests/second_source_i_2"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, s_source_decs, s_source_Ks, s_source_index = pickle.load(f)
        

    for i in range(len(s_source_decs)):
        for j in range(len(s_source_Ks)):
            
            temp_path = f"{data_folder}/{i}_{j}"

            pointings = PointingClusters(
                (temp_path,),
                min_angle_dif=1.5,
                max_angle_dif=7.5,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
                center_ra=10.,
                center_dec=-40.,
            ).pointings
            save_clusters(pointings, temp_path)

            pointings = load_clusters(temp_path)
            source_model = define_sources((
                (simulated_pl_0374, (200,)),
            ))

            multinest_fit = MultinestClusterFit(
                pointings,
                source_model,
                (None, None),
                np.geomspace(18, 3000, 50),
                log_binning_function_for_x_number_of_bins(125),
                # true_values=true_values(),
                folder=temp_path,
            )

            multinest_fit.parameter_fit_distribution()
            multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
            # multinest_fit.ppc()
            
            # print(multinest_fit._cc._all_parameters)
            
            p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
            val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
            cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

            with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
                pickle.dump((val, cov), f)                

def fit_second_source_i_3():
    data_folder = "./main_files/pure_simulation_tests/second_source_i_3"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, s_source_decs, s_source_Ks, s_source_index = pickle.load(f)
        

    for i in range(len(s_source_decs)):
        for j in range(len(s_source_Ks)):
            
            temp_path = f"{data_folder}/{i}_{j}"

            pointings = PointingClusters(
                (temp_path,),
                min_angle_dif=1.5,
                max_angle_dif=7.5,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(2,2),
                center_ra=10.,
                center_dec=-40.,
            ).pointings
            save_clusters(pointings, temp_path)

            pointings = load_clusters(temp_path)
            source_model = define_sources((
                (simulated_pl_0374, (200,)),
            ))

            multinest_fit = MultinestClusterFit(
                pointings,
                source_model,
                (None, None),
                np.geomspace(18, 3000, 50),
                log_binning_function_for_x_number_of_bins(125),
                # true_values=true_values(),
                folder=temp_path,
            )

            multinest_fit.parameter_fit_distribution()
            multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
            # multinest_fit.ppc()
            
            # print(multinest_fit._cc._all_parameters)
            
            p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
            val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
            cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

            with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
                pickle.dump((val, cov), f)

def fit_energy_range():
    data_folder = "./main_files/pure_simulation_tests/energy_range"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices, energy_ranges = pickle.load(f)
        
    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.5,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=300.,
        cluster_size_range=(2,2),
        center_ra=10.,
        center_dec=-40.,
    ).pointings
    save_clusters(pointings, data_folder)

    pointings = load_clusters(data_folder)
    source_model = define_sources((
        (simulated_pl_0374, (200,)),
    ))

    for i in range(len(energy_ranges)):
            
        temp_path = f"{data_folder}/{i}"

        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            energy_ranges[i],
            np.geomspace(18, 3000, 50),
            log_binning_function_for_x_number_of_bins(125),
            # true_values=true_values(),
            folder=temp_path,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()
        
        # print(multinest_fit._cc._all_parameters)
        
        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)

def fit_num_e_bins():
    data_folder = "./main_files/pure_simulation_tests/num_e_bins"
    
    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices, num_bins = pickle.load(f)
        
    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.5,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=300.,
        cluster_size_range=(2,2),
        center_ra=10.,
        center_dec=-40.,
    ).pointings
    save_clusters(pointings, data_folder)

    pointings = load_clusters(data_folder)
    source_model = define_sources((
        (simulated_pl_0374, (200,)),
    ))

    for i in range(len(num_bins)):
            
        temp_path = f"{data_folder}/{i}"

        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (None, None),
            np.geomspace(18, 3000, 50),
            log_binning_function_for_x_number_of_bins(num_bins[i]),
            # true_values=true_values(),
            folder=temp_path,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()
        
        # print(multinest_fit._cc._all_parameters)
        
        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)

def fit_data_scaling():
    data_folder = "./main_files/pure_simulation_tests/data_scaling"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices, time_fraction = pickle.load(f)

    for i in range(len(time_fraction)):
            
        temp_path = f"{data_folder}/t_{i}"
        
        pointings = PointingClusters(
            (temp_path,),
            min_angle_dif=1.5,
            max_angle_dif=7.5,
            max_time_dif=0.2,
            radius_around_source=10.,
            min_time_elapsed=10.,
            cluster_size_range=(2,2),
            center_ra=10.,
            center_dec=-40.,
        ).pointings
        save_clusters(pointings, temp_path)
        
        pointings = load_clusters(temp_path)
        source_model = define_sources((
            (simulated_pl_0374, (200,)),
        ))

        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (None, None),
            np.geomspace(18, 3000, 50),
            log_binning_function_for_x_number_of_bins(125),
            # true_values=true_values(),
            folder=temp_path,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()
        
        # print(multinest_fit._cc._all_parameters)
        
        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)

    # number of pointing pairs
    for i in range(len(time_fraction)):
            
        temp_path = f"{data_folder}/p_{i}"
        
        pointings_path = f"{data_folder}/t_0"
        
        pointings = load_clusters(pointings_path)
        num_pairs = int(len(pointings)*time_fraction[i])
        pointings = pointings[:num_pairs]
        source_model = define_sources((
            (simulated_pl_0374, (200,)),
        ))

        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (None, None),
            np.geomspace(18, 3000, 50),
            log_binning_function_for_x_number_of_bins(125),
            # true_values=true_values(),
            folder=temp_path,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()
        
        # print(multinest_fit._cc._all_parameters)
        
        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)

def fit_cluster_size():
    data_folder = "./main_files/pure_simulation_tests/cluster_size"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices, cluster_sizes = pickle.load(f)
        
    cluster_sizes = cluster_sizes[:-1]

    repeats = 3

    for i in range(len(cluster_sizes)):
        for j in range(repeats):
            temp_path = f"{data_folder}/{i}_{j}"
            
            pointings = PointingClusters(
                (data_folder,),
                min_angle_dif=1.5,
                max_angle_dif=7.5,
                max_time_dif=0.2,
                radius_around_source=10.,
                min_time_elapsed=300.,
                cluster_size_range=(cluster_sizes[i],cluster_sizes[i]),
                center_ra=10.,
                center_dec=-40.,
            ).pointings
            save_clusters(pointings, temp_path)

            pointings = load_clusters(temp_path)
            source_model = define_sources((
                (simulated_pl_0374, (200,)),
            ))

            multinest_fit = MultinestClusterFit(
                pointings,
                source_model,
                (None, None),
                np.geomspace(18, 3000, 50),
                log_binning_function_for_x_number_of_bins(125),
                # true_values=true_values(),
                folder=temp_path,
            )

            multinest_fit.parameter_fit_distribution()
            multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
            # multinest_fit.ppc()
            
            # print(multinest_fit._cc._all_parameters)
            
            p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
            val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
            cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

            with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
                pickle.dump((val, cov), f)

def fit_identical_new_gen():
    data_folder = "./main_files/pure_simulation_tests/identical_repeats_new_gen8"

    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices, repeats = pickle.load(f)
        
    
    source_model = define_sources((
        (simulated_pl_0374, (200,)),
    ))

    for i in range(repeats):
        
        temp_path = f"{data_folder}/{i}"
        
        
        pointings = PointingClusters(
            (temp_path,),
            min_angle_dif=2.5,
            max_angle_dif=20.,
            max_time_dif=0.2,
            radius_around_source=10.,
            min_time_elapsed=300.,
            cluster_size_range=(2,2),
            center_ra=10.,
            center_dec=-40.,
        ).pointings
        save_clusters(pointings, data_folder)

        pointings = load_clusters(data_folder)
            
        

        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (None, None),
            np.geomspace(18, 3000, 75),
            log_binning_function_for_x_number_of_bins(100),
            # true_values=true_values(),
            folder=temp_path,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()
        
        # print(multinest_fit._cc._all_parameters)
        
        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)
            
def fit_source_position():
    data_folder = "./main_files/pure_simulation_tests/source_position"


    with open(f"{data_folder}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index = pickle.load(f)
        
    pointings = PointingClusters(
        (data_folder,),
        min_angle_dif=1.5,
        max_angle_dif=7.5,
        max_time_dif=0.2,
        radius_around_source=10.,
        min_time_elapsed=300.,
        cluster_size_range=(2,2),
        center_ra=10.,
        center_dec=-40.,
    ).pointings
    save_clusters(pointings, data_folder)

    pointings = load_clusters(data_folder)
    source_model = define_sources((
        (simulated_pl_0374_free_pos, (200,)),
    ))
            
    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (None, None),
        np.geomspace(18, 2500, 40),
        log_binning_function_for_x_number_of_bins(25),
        # true_values=true_values(),
        folder=data_folder,
        parameter_names=["RA", "DEC", "K", "index"]
    )

    multinest_fit.parameter_fit_distribution([source_ra, source_dec, source_K, source_index])
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()
    
    # print(multinest_fit._cc._all_parameters)
    
    # p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
    # val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    # cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    # with open(f"{data_folder}/source_parameters.pickle", "wb") as f:
    #     pickle.dump((val, cov), f)


def residual_plt_test():
    data_folder = "./main_files/pure_simulation_tests/identical_repeats"




    pointings = load_clusters(data_folder)
    source_model = define_sources((
        (simulated_pl_0374, (200,)),
    ))

            
    temp_path = f"{data_folder}/test_res"

    multinest_fit = MultinestClusterFit(
        pointings,
        source_model,
        (None, None),
        np.geomspace(18, 3000, 50),
        log_binning_function_for_x_number_of_bins(125),
        # true_values=true_values(),
        folder=temp_path,
    )

    multinest_fit.parameter_fit_distribution()
    multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
    multinest_fit.ppc()
    
    # print(multinest_fit._cc._all_parameters)
    
    p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{temp_path}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)

residual_plt_test()