import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pickle
from PointingClusters import load_clusters

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

path = "./main_files/spimodfit_comparison_sim_source"



def combined_plot_0374_ppc_normal():
    l_path = f"{path}/pyspi_real_bkg/0374"
    
    pointing_path = f"{l_path}/pre_ppc"
    pointings = load_clusters(pointing_path)
    
    pointing_path_strict = f"{l_path}/post_ppc"
    pointings_strict = load_clusters(pointing_path_strict)
    
    real_vals = [6e-3, -2]
    
    pointings_good = [p for p in pointings if p in pointings_strict]
    pointings_bad = [p for p in pointings if not p in pointings_strict]
    
    fig, ax = plt.subplots()
    
    folders = [f"{l_path}/ind/{p[0][0]}_{p[1][0]}" for p in pointings_good]
    # d_M = []
    # for i in range(len(folders)):
    #     with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
    #         val, cov = pickle.load(f)
    #     d_M.append(mahalanobis_dist(val, cov, real_vals))
    names = ["Clusters with good PPC-CDFs" for i,p in enumerate(pointings_good)]
    edgecolors = ["C2" for i in range(len(pointings_good))]
    linestyles = ["solid" for i in range(len(pointings_good))]
    
    counter = 0
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        if counter == 0:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
        else:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, ls=linestyle)
        counter += 1
    
    
    
    folders = [f"{l_path}/ind/{p[0][0]}_{p[1][0]}" for p in pointings_bad]
    # d_M = []
    # for i in range(len(folders)):
    #     with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
    #         val, cov = pickle.load(f)
    #     d_M.append(mahalanobis_dist(val, cov, real_vals))
    names = ["Clusters with bad PPC-CDFs" for i,p in enumerate(pointings_bad)]
    edgecolors = ["C3" for i in range(len(pointings_bad))]
    linestyles = ["solid" for i in range(len(pointings_bad))]
    
    counter = 0
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        if counter == 0:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
        else:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, ls=linestyle)
        counter += 1
    
    
    
    folders = [f"{l_path}/pre_ppc", f"{l_path}/post_ppc"]
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    names = [f"Pre-PPC Combined Fit : $d_M={d_M[0]:.3f}$", f"Post-PPC Combined Fit : $d_M={d_M[1]:.3f}$"]
    edgecolors = ["C0", "C1"]
    linestyles = ["solid", "dashed"]
    lws = [4., 4.]
    
    
    for folder, name, edgecolor, linestyle, lw in zip(folders, names, edgecolors, linestyles, lws):
        with open(f"./{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle, lw=lw)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel("K [keV$^{-1}$s$^{-1}$cm$^{-2}$]")
    plt.ylabel("index")
    plt.title("$1.5^\circ$ Minimum Separation", fontsize=10, pad=1)
    plt.savefig(f"{path}/combined_plot_0374_ppc_normal.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

def combined_plot_0374_ppc_far():
    l_path = f"{path}/pyspi_real_bkg/0374"
    
    pointing_path = f"{l_path}/pre_ppc_far"
    pointings = load_clusters(pointing_path)
    
    pointing_path_strict = f"{l_path}/post_ppc_far"
    pointings_strict = load_clusters(pointing_path_strict)
    
    real_vals = [6e-3, -2]
    
    pointings_good = [p for p in pointings if p in pointings_strict]
    pointings_bad = [p for p in pointings if not p in pointings_strict]
    
    fig, ax = plt.subplots()
    
    folders = [f"{l_path}/ind_far/{p[0][0]}_{p[1][0]}" for p in pointings_good]
    # d_M = []
    # for i in range(len(folders)):
    #     with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
    #         val, cov = pickle.load(f)
    #     d_M.append(mahalanobis_dist(val, cov, real_vals))
    names = ["Clusters with good PPC-CDFs" for i,p in enumerate(pointings_good)]
    edgecolors = ["C2" for i in range(len(pointings_good))]
    linestyles = ["solid" for i in range(len(pointings_good))]
    
    counter = 0
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        if counter == 0:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
        else:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, ls=linestyle)
        counter += 1
    
    
    
    folders = [f"{l_path}/ind_far/{p[0][0]}_{p[1][0]}" for p in pointings_bad]
    # d_M = []
    # for i in range(len(folders)):
    #     with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
    #         val, cov = pickle.load(f)
    #     d_M.append(mahalanobis_dist(val, cov, real_vals))
    names = ["Clusters with bad PPC-CDFs" for i,p in enumerate(pointings_bad)]
    edgecolors = ["C3" for i in range(len(pointings_bad))]
    linestyles = ["solid" for i in range(len(pointings_bad))]
    
    counter = 0
    for folder, name, edgecolor, linestyle in zip(folders, names, edgecolors, linestyles):
        if counter == 0:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle)
        else:
            with open(f"./{folder}/source_parameters.pickle", "rb") as f:
                val, cov = pickle.load(f)
            confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, ls=linestyle)
        counter += 1
    
    
    
    folders = [f"{l_path}/pre_ppc_far", f"{l_path}/post_ppc_far"]
    d_M = []
    for i in range(len(folders)):
        with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        d_M.append(mahalanobis_dist(val, cov, real_vals))
    names = [f"Pre-PPC Combined Fit : $d_M={d_M[0]:.3f}$", f"Post-PPC Combined Fit : $d_M={d_M[1]:.3f}$"]
    edgecolors = ["C0", "C1"]
    linestyles = ["solid", "solid"]
    lws = [4., 4.]
    
    
    for folder, name, edgecolor, linestyle, lw in zip(folders, names, edgecolors, linestyles, lws):
        with open(f"./{folder}/source_parameters.pickle", "rb") as f:
            val, cov = pickle.load(f)
        confidence_ellipse(val, cov, ax, 1, edgecolor=edgecolor, label=name, ls=linestyle, lw=lw)
    plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
    ax.autoscale()
    lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.xlabel("K [keV$^{-1}$s$^{-1}$cm$^{-2}$]")
    plt.ylabel("index")
    plt.title("$2.5^\circ$ Minimum Separation", fontsize=10, pad=1)
    plt.savefig(f"{path}/combined_plot_0374_ppc_far.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

combined_plot_0374_ppc_far()
combined_plot_0374_ppc_normal()







