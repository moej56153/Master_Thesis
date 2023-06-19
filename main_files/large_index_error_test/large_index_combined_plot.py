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

folders = [
    "./main_files/large_index_error_test/source_parameters_large_index_125_50.pickle",
    "./main_files/large_index_error_test/source_parameters_large_index_500_200.pickle",
    "./main_files/large_index_error_test/source_parameters_large_index_1000_400.pickle",
    "./main_files/large_index_error_test/source_parameters_large_index_1000_600.pickle",
]

real_vals = np.array([8e-4, -8])

d_M = []
for i in range(len(folders)):
    with open(f"{folders[i]}", "rb") as f:
        val, cov = pickle.load(f)
    d_M.append(mahalanobis_dist(val, cov, real_vals))

bins = [
    [125, 50],
    [500, 200],
    [1000, 400],
    [1000, 600]
]

names = [f"$N(E_{{in}})={bins[i][0]},N(E_{{out}})={bins[i][1]},d_M={d_M[i]:.3f}$" for i in range(len(d_M))]

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
    with open(f"./{folder}", "rb") as f:
        val, cov = pickle.load(f)
    confidence_ellipse(val, cov, ax, 10, edgecolor=edgecolor, label=name, ls=linestyle)
ax.autoscale()
plt.plot(real_vals[0], real_vals[1], "ko", label="True Values")
# lgd = plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
lgd = plt.legend()
plt.xlabel("K")
plt.ylabel("index")
plt.savefig("./main_files/large_index_error_test/large_index_combined_plot_10sig.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')