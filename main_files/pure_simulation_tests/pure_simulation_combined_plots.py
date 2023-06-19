import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pickle

def mahalanobis_dist(vals, cov, real_vals):
    dif = (vals - real_vals)
    return np.sqrt(np.linalg.multi_dot([dif, np.linalg.inv(cov), dif]))

def confidence_ellipse(val, cov, ax, n_std=1.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = val[0]

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = val[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

path = "./main_files/pure_simulation_tests"

def repeated_identical():
    l_path = f"{path}/identical_repeats"
    
    repeats = 8
    
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [f"{l_path}/{i}" for i in range(repeats)]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    
    

    names = [f"{i}: $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid"
    ]
    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel("K")
    plt.ylabel("index")
    plt.savefig(f"{path}/combined_plot_repeated_identical.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def repeated_identical_new_gen():
    l_path = f"{path}/identical_repeats_new_gen"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, repeats = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [f"{l_path}/{i}" for i in range(repeats)]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    
    

    names = [f"{i}: $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        # "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8"
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid"
    ]
    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    
    plt.xlabel("K")
    plt.ylabel("index")
    
    new_gens = 8
    l_paths = [f"{path}/identical_repeats_new_gen"] + [f"{path}/identical_repeats_new_gen{i}" for i in range(2,new_gens+1)]
    with open(f"{l_paths[0]}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, repeats = pickle.load(f)
    real_vals = [source_K, source_index]
    folders = [f"{l_paths[i]}/{j}" for j in range(repeats) for i in range(new_gens)]
    d_M = []
    means = np.zeros((len(folders), 2))
    covs = np.zeros((len(folders), 2, 2))
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        means[i] = val
        covs[i] = cov * (len(folders)-1)/len(folders)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    mean = np.mean(means, axis=0)
    cov = np.sum(covs, axis=0)/(len(d_M)**2)
    plt.plot(mean[0], mean[1], "ro", label=f"Mean Values (N={len(d_M)})")
    
    samp_cov = np.cov(means[:,0], means[:,1]) / len(d_M)
    tot_cov = cov + samp_cov
    d_M_m = mahalanobis_dist(mean, tot_cov, real_vals)
    confidence_ellipse(mean, tot_cov, ax, 1, edgecolor="r", label=f"Mean Confidence Interval: $d_M = {d_M_m:.3f}$", ls="dashed")
    
    
    lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    
    
    plt.savefig(f"{path}/combined_plot_repeated_identical_new_gen.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def num_e_bins():
    l_path = f"{path}/num_e_bins"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, bin_number = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [f"{l_path}/{i}" for i in range(len(bin_number))]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    
    

    names = [f"$N_B$ = {bin_number[i]}: $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
    ]
    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    # lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    lgd = plt.legend()
    plt.xlabel("K")
    plt.ylabel("index")
    # fig.autofmt_xdate()
    plt.savefig(f"{path}/combined_plot_num_e_bins.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def energy_range():
    l_path = f"{path}/energy_range"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, energy_ranges = pickle.load(f)
                
    real_vals = [source_K, source_index]
    
    folders = [f"{l_path}/{i}" for i in range(len(energy_ranges))]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
        
        
    sigmas = [4, 4, 4, 4, 4, 1, 1, 1]
    
    new_e_range = []
    for i in energy_ranges:
        temp = []
        if i[0] is None:
            temp.append(18)
        else:
            temp.append(i[0])
        if i[1] is None:
            temp.append(2000)
        else:
            temp.append(i[1])
        new_e_range.append(temp)

    names = [f"{new_e_range[i][0]} - {new_e_range[i][1]} keV : $d_M$ = {d_M[i]:.3f} : {sigmas[i]}$\sigma$" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7"
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid"
    ]
    
    del folders[-2:]
    del names[-2:]
    del edgecolors[-2:]
    del linestyles[-2:]
    del sigmas[-2:]
    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle, sigma in zip(folders, names, edgecolors, linestyles, sigmas):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, sigma, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel("K")
    plt.ylabel("index")
    fig.autofmt_xdate()
    plt.savefig(f"{path}/combined_plot_energy_range.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def data_scaling():
    l_path = f"{path}/data_scaling"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, time_fraction = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [f"{l_path}/t_{i}" for i in range(3)] + [f"{l_path}/p_{i}" for i in range(3)]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    
    

    names = ["1.0t", "0.3t", "0.1t", "42SCWs", "12SCWs", "4SCWs"]

    names = [f"{names[i]} : $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "dashed",
        "dashed",
        "dashed",
    ]
    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    # lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    lgd = plt.legend()
    plt.xlabel("K")
    plt.ylabel("index")
    # fig.autofmt_xdate()
    plt.savefig(f"{path}/combined_plots_data_scaling.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def cluster_size():
    l_path = f"{path}/cluster_size"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, cluster_sizes = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [
        f"{l_path}/0_0",
        f"{l_path}/0_1",
        f"{l_path}/0_2",
        f"{l_path}/1_0",
        f"{l_path}/1_1",
        f"{l_path}/1_2",
    ]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    
    

    names = [f"$S_C$ = {cluster_sizes[i//3]}: $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "dashed",
        "dashed",
        "dashed",
    ]
    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    # lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    lgd = plt.legend()
    plt.xlabel("K")
    plt.ylabel("index")
    # fig.autofmt_xdate()
    plt.savefig(f"{path}/combined_plots_cluster_size.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def different_sources():
    l_path = f"{path}/different_sources"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_Ks, source_indices = pickle.load(f)
        
    real_vals = np.array([
        [5e-5, -0.5],
        [5e-5, -2],
        [5e-5, -8],
        [2e-4, -0.5],
        [2e-4, -2],
        [2e-4, -8],
        [8e-4, -0.5],
        [8e-4, -2],
        [8e-4, -8]
    ])
    
    folders = [
        f"{l_path}/0_0",
        f"{l_path}/0_1",
        f"{l_path}/0_2",
        f"{l_path}/1_0",
        f"{l_path}/1_1",
        f"{l_path}/1_2",
        f"{l_path}/2_0",
        f"{l_path}/2_1",
        f"{l_path}/2_2",
    ]
    
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals[i]))
    
    

    names = [f"$d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
    ]
    
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
    fig.tight_layout(h_pad=1, w_pad=2.5)
    for folder, name, edgecolor, linestyle, counter in zip(folders, names, edgecolors, linestyles, range(len(folders))):
        axis = ax[counter//3, counter%3]
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, axis, 1, edgecolor=edgecolor, label=name, ls=linestyle)
        
        
        
        
        if not counter%3 == 2:
            axis.plot(real_vals[counter,0], real_vals[counter,1], "ko", label="True Values")
        else:
            axis.plot(val[0], val[1], "wo", label=f"True Values={real_vals[counter,0]:.0e}, {real_vals[counter,1]}")
        lgd = axis.legend()
        axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        axis.autoscale()

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("K", labelpad=20)
    plt.ylabel("index", labelpad=28)
    # fig.autofmt_xdate()
    plt.savefig(f"{path}/combined_plots_different_sources.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def second_source():
    l_path = f"{path}"
        
        
    folders1 = [
        f"{l_path}/second_source_i_1/0_0",
        f"{l_path}/second_source_i_1/0_1",
        f"{l_path}/second_source_i_1/0_2",
        f"{l_path}/second_source_i_2/0_0",
        f"{l_path}/second_source_i_2/0_1",
        f"{l_path}/second_source_i_2/0_2",
        f"{l_path}/second_source_i_3/0_0",
        f"{l_path}/second_source_i_3/0_1",
        f"{l_path}/second_source_i_3/0_2",
    ]
    folders2 = [
        f"{l_path}/second_source_i_1/1_0",
        f"{l_path}/second_source_i_1/1_1",
        f"{l_path}/second_source_i_1/1_2",
        f"{l_path}/second_source_i_2/1_0",
        f"{l_path}/second_source_i_2/1_1",
        f"{l_path}/second_source_i_2/1_2",
        f"{l_path}/second_source_i_3/1_0",
        f"{l_path}/second_source_i_3/1_1",
        f"{l_path}/second_source_i_3/1_2",
    ]
    folders3 = [
        f"{l_path}/second_source_i_1/2_0",
        f"{l_path}/second_source_i_1/2_1",
        f"{l_path}/second_source_i_1/2_2",
        f"{l_path}/second_source_i_2/2_0",
        f"{l_path}/second_source_i_2/2_1",
        f"{l_path}/second_source_i_2/2_2",
        f"{l_path}/second_source_i_3/2_0",
        f"{l_path}/second_source_i_3/2_1",
        f"{l_path}/second_source_i_3/2_2",
    ]
    folders4 = [
        f"{l_path}/second_source_i_1/3_0",
        f"{l_path}/second_source_i_1/3_1",
        f"{l_path}/second_source_i_1/3_2",
        f"{l_path}/second_source_i_2/3_0",
        f"{l_path}/second_source_i_2/3_1",
        f"{l_path}/second_source_i_2/3_2",
        f"{l_path}/second_source_i_3/3_0",
        f"{l_path}/second_source_i_3/3_1",
        f"{l_path}/second_source_i_3/3_2",
    ]
    folders5 = [
        f"{l_path}/second_source_i_1/4_0",
        f"{l_path}/second_source_i_1/4_1",
        f"{l_path}/second_source_i_1/4_2",
        f"{l_path}/second_source_i_2/4_0",
        f"{l_path}/second_source_i_2/4_1",
        f"{l_path}/second_source_i_2/4_2",
        f"{l_path}/second_source_i_3/4_0",
        f"{l_path}/second_source_i_3/4_1",
        f"{l_path}/second_source_i_3/4_2",
    ]
    folders6 = [
        f"{l_path}/second_source_i_1/5_0",
        f"{l_path}/second_source_i_1/5_1",
        f"{l_path}/second_source_i_1/5_2",
        f"{l_path}/second_source_i_2/5_0",
        f"{l_path}/second_source_i_2/5_1",
        f"{l_path}/second_source_i_2/5_2",
        f"{l_path}/second_source_i_3/5_0",
        f"{l_path}/second_source_i_3/5_1",
        f"{l_path}/second_source_i_3/5_2",
    ]
    t_folders = (folders1, folders2, folders3, folders4, folders5, folders6)
    
    real_vals = np.array([1e-4,-2])
    
    
    d_M1 = []
    for i in range(len(folders1)):
        with open(f"{folders1[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M1.append(mahalanobis_dist(val, cov, real_vals))
        
    d_M2 = []
    for i in range(len(folders2)):
        with open(f"{folders2[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M2.append(mahalanobis_dist(val, cov, real_vals))
        
    d_M3 = []
    for i in range(len(folders3)):
        with open(f"{folders3[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M3.append(mahalanobis_dist(val, cov, real_vals))
        
    d_M4 = []
    for i in range(len(folders4)):
        with open(f"{folders4[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M4.append(mahalanobis_dist(val, cov, real_vals))
        
    d_M5 = []
    for i in range(len(folders5)):
        with open(f"{folders5[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M5.append(mahalanobis_dist(val, cov, real_vals))
        
    d_M6 = []
    for i in range(len(folders6)):
        with open(f"{folders6[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M6.append(mahalanobis_dist(val, cov, real_vals))
        
    t_d_M = (d_M1, d_M2, d_M3, d_M4, d_M5, d_M6)
    
    

    names1 = [f"$d_M$ = {d_M1[i]:.3f}" for i in range(len(d_M1))]
    names2 = [f"$d_M$ = {d_M2[i]:.3f}" for i in range(len(d_M2))]
    names3 = [f"$d_M$ = {d_M3[i]:.3f}" for i in range(len(d_M3))]
    names4 = [f"$d_M$ = {d_M4[i]:.3f}" for i in range(len(d_M4))]
    names5 = [f"$d_M$ = {d_M5[i]:.3f}" for i in range(len(d_M5))]
    names6 = [f"$d_M$ = {d_M6[i]:.3f}" for i in range(len(d_M6))]

    t_names = (names1, names2, names3, names4, names5, names6)
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
    ]
    linestyles = [
        "dotted",
        "dotted",
        "dotted",
        "dashed",
        "dashed",
        "dashed",
        "solid",
        "solid",
        "solid",
    ]
    
    sd = [10, 10, 4, 2, 1, 1]
    ang_difs = [0, 0.5, 1.5, 5, 15, 45]
    
    labels = [
        [0.1e-4, -1],
        [0.3e-4, -1],
        [1e-4, -1],
        [0.1e-4, -2],
        [0.3e-4, -2],
        [1e-4, -2],
        [0.1e-4, -3],
        [0.3e-4, -3],
        [1e-4, -3],
    ]
    
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,12))
    fig.tight_layout(h_pad=2, w_pad=12)
    for plot, folders, names, d_M in zip(range(6), t_folders, t_names, t_d_M):
        axis = ax[plot//2, plot%2]
        for folder, name, edgecolor, linestyle, counter in zip(folders, names, edgecolors, linestyles, range(9)):
            with open(f"{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, axis, sd[plot], edgecolor=edgecolor, label=name, ls=linestyle)
            
            
            axis.autoscale()
            

        axis.plot(real_vals[0], real_vals[1], "ko", label="True Values")
        axis.set_title(f"{ang_difs[plot]}$^\circ$   {sd[plot]}$\sigma$")
        lgda = axis.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
        axis.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel("K", labelpad=20, fontsize=16)
    plt.ylabel("index", labelpad=16, fontsize=16)


    import matplotlib.patches as mpatches

    patch = mpatches.Patch(color='C0', label='Manual Label')

    handles = [mpatches.Patch(color=f'C{i}', label=f'K={labels[i][0]:.0e}, index={labels[i][1]}') for i in range(9)]

    lgdp = plt.legend(handles=handles, bbox_to_anchor=(1, -0.05), loc=1, borderaxespad=0.)
    plt.savefig(f"{path}/combined_plots_second_source.pdf", bbox_extra_artists=(lgdp,lgda), bbox_inches='tight')

def d_M_distribution():
    def prob_calc(x1, x2):
        return np.exp(-1/2 * x1**2) - np.exp(-1/2 * x2**2)
    
    new_gens = 8
    
    l_paths = [f"{path}/identical_repeats_new_gen"] + [f"{path}/identical_repeats_new_gen{i}" for i in range(2,new_gens+1)]
    
    
    with open(f"{l_paths[0]}/source_params.pickle", "rb") as f:
        source_ra, source_dec, source_piv, source_K, source_index, repeats = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [f"{l_paths[i]}/{j}" for j in range(repeats) for i in range(new_gens)]
    
    d_M = []
    means = np.zeros((len(folders), 2))
    covs = np.zeros((len(folders), 2, 2))
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        means[i] = val
        covs[i] = cov * (len(folders)-1)/len(folders)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
        
    mean = np.mean(means, axis=0)
    d_M2 = []
    print(mean)
    for i in range(len(folders)):
        d_M2.append(mahalanobis_dist(means[i], covs[i], mean))
        
    bins = np.linspace(0, max(max(d_M), max(d_M2)), 6)
    hists, _ = np.histogram(d_M, bins)
    hists = hists / len(d_M)
    
    hists2, _ = np.histogram(d_M2, bins)
    hists2 = hists2 / len(d_M2)
    
    fig, ax = plt.subplots()
    
    probs = prob_calc(bins[:-1], bins[1:])
    ax.stairs(probs, bins, label="Expected Distribution", lw=10.5, color="C7")
    
    ax.stairs(hists, bins, label=f"Observed Distribution around True Values (N={len(d_M)})", lw=7., color="C0")
    ax.stairs(hists2, bins, label=f"Observed Distribution around Mean Values (N={len(d_M)})", lw=3.5, color="C1")
    
    
    
    
    
    ax.set_xlabel("Mahalanobis Distance")
    ax.set_ylabel("Probability")
    
    lgd = ax.legend(bbox_to_anchor=(1.0, 1.2), loc=1, borderaxespad=0.)
    
    fig.savefig(f"{path}/d_M_distribution.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def pointing_distances():
    l_path = f"{path}/pointing_distance"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_piv, source_K, source_index, pointing_distances = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [f"{l_path}/{i}" for i in range(len(pointing_distances))]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    
    

    names = [f"$d_P$ = {2*pointing_distances[i]:.1f} : $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "C8",
        "C9"
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
        "solid"
    ]
    
    del d_M[0]
    del folders[0]
    del names[0]
    del edgecolors[0]
    del linestyles[0]
    
    del d_M[4]
    del folders[4]
    del names[4]
    del edgecolors[4]
    del linestyles[4]
    
    del d_M[6]
    del folders[6]
    del names[6]
    del edgecolors[6]
    del linestyles[6]
    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel("K")
    plt.ylabel("index")
    fig.autofmt_xdate()
    plt.savefig(f"{path}/combined_plot_pointing_distances.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def source_distances():
    l_path = f"{path}/source_distance"
        
    with open(f"{l_path}/source_params.pickle", "rb") as f:
        source_piv, source_K, source_index, source_distances = pickle.load(f)
        
    real_vals = [source_K, source_index]
    
    folders = [f"{l_path}/{i}" for i in range(len(source_distances))]
    
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    
    

    names = [f"$d_S$ = {source_distances[i]:.1f} : $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]
    
    edgecolors = [
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
    ]
    linestyles = [
        "solid",
        "solid",
        "solid",
        "solid",
        "solid",
    ]
    
    del d_M[-1]
    del folders[-1]
    del names[-1]
    del edgecolors[-1]
    del linestyles[-1]
    

    
    fig, ax = plt.subplots()
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        with open(f"{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel("K")
    plt.ylabel("index")
    fig.autofmt_xdate()
    plt.savefig(f"{path}/combined_plot_source_distances.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

repeated_identical_new_gen()
d_M_distribution()