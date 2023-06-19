import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pickle

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


folders = [
    "./main_files/ppc_test/ppc_test_1662_simple",
    "./main_files/ppc_test/ppc_test_1664_simple",
    "./main_files/ppc_test/ppc_test_1667_simple",
    "./main_files/ppc_test/ppc_test_1662_simple_wo_outliers",
    "./main_files/ppc_test/ppc_test_1664_simple_wo_outliers",
    "./main_files/ppc_test/ppc_test_1667_simple_wo_outliers",
]
names = [
    "1662 with outliers",
    "1664 with outliers",
    "1667 with outliers",
    "1662 without outliers",
    "1664 without outliers",
    "1667 without outliers",
]
edgecolors = [
    "C1",
    "C2",
    "C3",
    "C1",
    "C2",
    "C3",
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
    with open(f"./{folder}/source_parameters.pickle", "rb") as f:
        val, cov = pickle.load(f)
    confidence_ellipse(val, cov, ax, 3, edgecolor=edgecolor, label=name, ls=linestyle)
ax.autoscale()
lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.xlabel("Crab K")
plt.ylabel("Crab index")
plt.savefig("./main_files/ppc_test/combined_plot_3sig.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')