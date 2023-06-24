import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
from MultinestClusterFit import MultinestClusterFit
from RebinningFunctions import spimodfit_binning_SE, log_binning_function_for_x_number_of_bins, no_rebinning #, rebin_data_exp_50
from PointingClusters import PointingClusters, save_clusters, load_clusters
from ModelSources import *
import pickle



# ra, dec = 155., 75.

def pyspi_real_bkg_fit_0374_pre_ppc():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    # fit_path = f"{data_path}/pre_ppc_far"
    fit_path = f"{data_path}/pre_ppc"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    pointings = PointingClusters(
            (data_path,),
            # min_angle_dif=2.5,
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
    
    # pointings = load_clusters(data_path)
    
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

def pyspi_real_bkg_fit_0374_pre_ppc_far():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    fit_path = f"{data_path}/pre_ppc_far"
    # fit_path = f"{data_path}/pre_ppc"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    pointings = PointingClusters(
            (data_path,),
            min_angle_dif=2.5,
            # min_angle_dif=1.5,
            max_angle_dif=10.,
            max_time_dif=0.2,
            radius_around_source=10.,
            min_time_elapsed=300.,
            cluster_size_range=(2,2),
            center_ra=ra,
            center_dec=dec,
        ).pointings
    save_clusters(pointings, fit_path)
    
    # pointings = load_clusters(data_path)
    
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


def pyspi_real_bkg_fit_0374_post_ppc():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    pointings_path = f"{data_path}/pre_ppc"
    fit_path = f"{data_path}/post_ppc"
    # pointings_path = f"{data_path}/pre_ppc_far"
    # fit_path = f"{data_path}/post_ppc_far"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    
    pointings2 = load_clusters(pointings_path)

    pointings = []

    bad_pointings = (
        "037400020010",
        # "037400160010",
        # "037400440010",
        )
    


    for cluster in pointings2:
        if cluster[0][0] in bad_pointings:
            continue
        else:
            pointings.append(cluster)
            
    pointings = tuple(pointings)
    
    save_clusters(pointings, fit_path)
    
    # return 0
    
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
    # multinest_fit.ppc()
    
    p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{fit_path}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)

def pyspi_real_bkg_fit_0374_post_ppc_far():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    # pointings_path = f"{data_path}/pre_ppc"
    # fit_path = f"{data_path}/post_ppc"
    pointings_path = f"{data_path}/pre_ppc_far"
    fit_path = f"{data_path}/post_ppc_far"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    
    pointings2 = load_clusters(pointings_path)

    pointings = []


    
    bad_pointings = (
        "037400020010",
        "037400030010",
        "037400140010",
        "037400150010",
        # "037400230010",
        # "037400240010",
        # "037400430010",
        )

    for cluster in pointings2:
        if cluster[0][0] in bad_pointings:
            continue
        else:
            pointings.append(cluster)
            
    pointings = tuple(pointings)
    
    save_clusters(pointings, fit_path)
    
    # return 0
    
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
    # multinest_fit.ppc()
    
    p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{fit_path}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)




def pyspi_real_bkg_fit_0374_pre_ppc_triple():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    fit_path = f"{data_path}/pre_ppc_triple"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    pointings = PointingClusters(
            (data_path,),
            min_angle_dif=1.5,
            max_angle_dif=10.,
            max_time_dif=0.2,
            radius_around_source=10.,
            min_time_elapsed=300.,
            cluster_size_range=(3,3),
            center_ra=ra,
            center_dec=dec,
        ).pointings
    save_clusters(pointings, fit_path)
    
    # pointings = load_clusters(data_path)
    
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

def pyspi_real_bkg_fit_0374_post_ppc_triple():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    pointings_path = f"{data_path}/pre_ppc_triple"
    fit_path = f"{data_path}/post_ppc_triple"

    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    
    pointings2 = load_clusters(pointings_path)

    pointings = []

    bad_pointings = (
        "037400020010",
        "037400140010",
        "037400430010",
        )
    

    for cluster in pointings2:
        if cluster[0][0] in bad_pointings:
            continue
        else:
            pointings.append(cluster)
            
    pointings = tuple(pointings)
    
    save_clusters(pointings, fit_path)
    
    # return 0
    
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

def pyspi_smf_bkg_fit_0374_pre_ppc():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_smf_bkg/{rev}"
    fit_path = f"{data_path}/pre_ppc"
    
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
    
    # pointings = load_clusters(data_path)
    
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
        # log_binning_function_for_x_number_of_bins(70),
        no_rebinning,
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

def pyspi_smf_bkg_fit_0374_post_ppc():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_smf_bkg/{rev}"
    pointings_path = f"{data_path}/pre_ppc"
    fit_path = f"{data_path}/post_ppc"

    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    
    pointings2 = load_clusters(pointings_path)

    pointings = []

    bad_pointings = (
        # "037400020010",
        # "037400230010",
        # "037400270010",
        # "037400330010",
        "037400400010",
        # "037400430010",
        )
    

    for cluster in pointings2:
        if cluster[0][0] in bad_pointings:
            continue
        else:
            pointings.append(cluster)
            
    pointings = tuple(pointings)
    
    save_clusters(pointings, fit_path)
    
    # return 0
    
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
    # multinest_fit.ppc()
    
    p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
    val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
    cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

    with open(f"{fit_path}/source_parameters.pickle", "wb") as f:
        pickle.dump((val, cov), f)


def pyspi_const_bkg_fit_0374_pre_ppc():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_const_bkg/{rev}"
    fit_path = f"{data_path}/pre_ppc"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    # pointings = PointingClusters(
    #         (data_path,),
    #         min_angle_dif=1.5,
    #         max_angle_dif=10.,
    #         max_time_dif=0.2,
    #         radius_around_source=10.,
    #         min_time_elapsed=300.,
    #         cluster_size_range=(2,2),
    #         center_ra=ra,
    #         center_dec=dec,
    #     ).pointings
    # save_clusters(pointings, fit_path)
    
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
        # log_binning_function_for_x_number_of_bins(70),
        spimodfit_binning_SE,
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



def pyspi_real_bkg_fit_0374_ind():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    fit_path = f"{data_path}/ind"
    pointing_path = f"{data_path}/pre_ppc"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    pointings2 = load_clusters(pointing_path)
    
    if rev=="0374":
        s = simulated_pl_0374
    elif rev=="1380":
        s = simulated_pl_1380
    
    source_model = define_sources((
        (s, (40,)),
    ))
    
    for i in range(len(pointings2)):
        pointings = (pointings2[i],)
        folder = f"{fit_path}/{pointings[0][0][0]}_{pointings[0][1][0]}"
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (30., 400.,),
            np.geomspace(18, 600, 50),
            log_binning_function_for_x_number_of_bins(70),
            # true_values=true_values(),
            folder=folder,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()

        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{folder}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)   

def pyspi_real_bkg_fit_0374_far_ind():
    rev = "0374"
    ra, dec = 10, -40
    data_path = f"./main_files/spimodfit_comparison_sim_source/pyspi_real_bkg/{rev}"
    fit_path = f"{data_path}/ind_far"
    pointing_path = f"{data_path}/pre_ppc_far"
    
    if not os.path.exists(f"{fit_path}"):
        os.mkdir(f"{fit_path}")
    
    pointings2 = load_clusters(pointing_path)
    
    if rev=="0374":
        s = simulated_pl_0374
    elif rev=="1380":
        s = simulated_pl_1380
    
    source_model = define_sources((
        (s, (40,)),
    ))
    
    for i in range(len(pointings2)):
        pointings = (pointings2[i],)
        folder = f"{fit_path}/{pointings[0][0][0]}_{pointings[0][1][0]}"
        
        multinest_fit = MultinestClusterFit(
            pointings,
            source_model,
            (30., 400.,),
            np.geomspace(18, 600, 50),
            log_binning_function_for_x_number_of_bins(70),
            # true_values=true_values(),
            folder=folder,
        )

        multinest_fit.parameter_fit_distribution()
        multinest_fit.text_summaries(pointing_combinations=True, reference_values=False, parameter_fit_constraints=False)
        # multinest_fit.ppc()

        p = ["Simulated Source 0374 K", "Simulated Source 0374 index"]
        val = np.array([i[1] for i in multinest_fit._cc.analysis.get_summary(parameters=p).values()])
        cov = multinest_fit._cc.analysis.get_covariance(parameters=p)[1]

        with open(f"{folder}/source_parameters.pickle", "wb") as f:
            pickle.dump((val, cov), f)  





pyspi_smf_bkg_fit_0374_pre_ppc()
pyspi_const_bkg_fit_0374_pre_ppc()


