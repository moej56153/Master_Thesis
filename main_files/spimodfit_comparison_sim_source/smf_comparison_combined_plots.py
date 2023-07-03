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

path = "./main_files/spimodfit_comparison_sim_source"


    
real_vals = [6e-3, -2]

folders = [
    f"{path}/pyspi_real_bkg/0374/pre_ppc",
    # f"{path}/pyspi_real_bkg/0374/post_ppc",
    f"{path}/pyspi_smf_bkg/0374/pre_ppc",
    # f"{path}/pyspi_smf_bkg/0374/post_ppc",
    f"{path}/pyspi_const_bkg/0374/pre_ppc",
    
    f"{path}/spimodfit_fits/0374_real_bkg",
    f"{path}/spimodfit_fits/0374_smf_bkg",
]

d_M = []
for i in range(len(folders)):
    with open(f"{folders[i]}/source_parameters.pickle", "rb") as f:
        val, cov = pickle.load(f)
    d_M.append(mahalanobis_dist(val, cov, real_vals))

names = [
    "PySpi Real Bkg",
    # "PySpi Real Bkg Post-PPC",
    "PySpi SMF Bkg",
    # "PySpi SMF Bkg Post-PPC",
    "PySpi Const Bkg",
    "Spimodfit Real Bkg",
    "Spimodfit SMF Bkg"
]

names = [f"{names[i]}: $d_M$ = {d_M[i]:.3f}" for i in range(len(d_M))]

edgecolors = [
    "C0",
    # "C0",
    "C1",
    # "C1",
    "C2",
    
    "C0",
    "C1",
]
linestyles = [
    "solid",
    # "dotted",
    "solid",
    # "dotted",
    "solid",
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
lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.xlabel("K [keV$^{-1}$s$^{-1}$cm$^{-2}$]")
plt.ylabel("index")
plt.savefig(f"{path}/spimodfit_comparison_combined_plot.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')