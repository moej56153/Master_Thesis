import sys, os
sys.path.insert(0, os.path.abspath('./main_files'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numba import njit
from MultinestClusterFit import *


input_b = np.geomspace(0.1,100,20)
input_s1 = np.geomspace(0.01,10,20)
input_t1 = np.geomspace(600, 6000, 20)
input_s2 = np.geomspace(0.01,10,20)
input_t2 = np.geomspace(600, 6000, 20)




def scaling_beviour_similar_sources():
    v_b = calc_bmaxL_variance_matrix(
        np.array(input_b),
        np.array([input_s1[10]]),
        np.array([input_t1[10]]),
        np.array([input_s2[10]]),
        np.array([input_t2[10]])
    )

    v_s1 = calc_bmaxL_variance_matrix(
        np.array([input_b[10]]),
        np.array(input_s1),
        np.array([input_t1[10]]),
        np.array([input_s2[10]]),
        np.array([input_t2[10]])
    )

    v_t1 = calc_bmaxL_variance_matrix(
        np.array([input_b[10]]),
        np.array([input_s1[10]]),
        np.array(input_t1),
        np.array([input_s2[10]]),
        np.array([input_t2[10]])
    )

    v_s2 = calc_bmaxL_variance_matrix(
        np.array([input_b[10]]),
        np.array([input_s1[10]]),
        np.array([input_t1[10]]),
        np.array(input_s2),
        np.array([input_t2[10]])
    )

    v_t2 = calc_bmaxL_variance_matrix(
        np.array([input_b[10]]),
        np.array([input_s1[10]]),
        np.array([input_t1[10]]),
        np.array([input_s2[10]]),
        np.array(input_t2)
    )
    
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12,7))
    fig.tight_layout()

    v_mats = (v_b, v_s1, v_t1, v_s2, v_t2)
    xs = (input_b, input_s1, input_t1, input_s2, input_t2)
    v_indices = ((0,0), (0,1), (1,1))

    b_int_funcs = (interpolate_linear, interpolate_linear, interpolate_logarithmic, interpolate_linear, interpolate_logarithmic)
    c_int_funcs = (interpolate_logarithmic, interpolate_linear, interpolate_powerlaw, interpolate_linear, interpolate_logarithmic)
    s_int_funcs = (interpolate_constant, interpolate_linear, interpolate_powerlaw, interpolate_constant, interpolate_constant)

    int_funcs = (b_int_funcs, c_int_funcs, s_int_funcs)

    axis_scales = (
        (("linear", "linear"), ("linear", "linear"), ("log", "linear"), ("linear", "linear"), ("log", "linear")),
        (("log", "linear"), ("linear", "linear"), ("log", "symlog"), ("linear", "linear"), ("log", "linear")),
        (("log", "linear"), ("linear", "linear"), ("log", "log"), ("log", "linear"), ("log", "linear"))
    )

    x_labels = ("b", "s1", "t1", "s2", "t2")
    y_labels = ("var($B_{d1}$)", "covar($B_{d1}$,$S_{d1}$)", "var($S_{d1}$)")

    labels = (
        (("lin"), ("lin"), ("log"), ("lin"), ("log")),
        (("log"), ("lin"), ("pow"), ("lin"), ("log")),
        (("con"), ("lin"), ("pow"), ("con"), ("con")),
    )

    for row in range(len(axes)):
        for col in range(len(axes[0])):
            x = xs[col]
            y = v_mats[col][:,:,:,:,:,0,v_indices[row][0],v_indices[row][1]].flatten()
            int_func = int_funcs[row][col]
            axes[row,col].plot(x, y, label="MC")
            y2 = int_func(x[0], x[-1], y[0], y[-1], x)
            axes[row,col].plot(x, y2, label=labels[row][col])
            
            axes[row,col].legend()
            
            axes[row,col].set_xscale(axis_scales[row][col][0])
            axes[row,col].set_yscale(axis_scales[row][col][1])
            
            if row==1 and col==2:
                axes[row,col].set_ylim(1.1*np.amin(y), 0.9*np.amax(y))
                
            if col==0:
                axes[row,col].set_ylabel(y_labels[row])
                
            if row==2:
                axes[row,col].set_xlabel(x_labels[col])
                
    plt.savefig("./main_files/ppc_test/PPC_scaling_similar.pdf", bbox_inches='tight')
    
def scaling_beviour_different_sources_1_bright_short():
    v_b = calc_bmaxL_variance_matrix(
        np.array(input_b),
        np.array([input_s1[19]]),
        np.array([input_t1[1]]),
        np.array([input_s2[2]]),
        np.array([input_t2[18]])
    )

    v_s1 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array(input_s1),
        np.array([input_t1[1]]),
        np.array([input_s2[2]]),
        np.array([input_t2[18]])
    )

    v_t1 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array([input_s1[19]]),
        np.array(input_t1),
        np.array([input_s2[2]]),
        np.array([input_t2[18]])
    )

    v_s2 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array([input_s1[19]]),
        np.array([input_t1[1]]),
        np.array(input_s2),
        np.array([input_t2[18]])
    )

    v_t2 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array([input_s1[19]]),
        np.array([input_t1[1]]),
        np.array([input_s2[2]]),
        np.array(input_t2)
    )
    
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12,7))
    fig.tight_layout()

    v_mats = (v_b, v_s1, v_t1, v_s2, v_t2)
    xs = (input_b, input_s1, input_t1, input_s2, input_t2)
    v_indices = ((0,0), (0,1), (1,1))

    b_int_funcs = (interpolate_linear, interpolate_linear, interpolate_logarithmic, interpolate_linear, interpolate_logarithmic)
    c_int_funcs = (interpolate_logarithmic, interpolate_linear, interpolate_powerlaw, interpolate_linear, interpolate_logarithmic)
    s_int_funcs = (interpolate_constant, interpolate_linear, interpolate_powerlaw, interpolate_constant, interpolate_constant)

    int_funcs = (b_int_funcs, c_int_funcs, s_int_funcs)

    axis_scales = (
        (("linear", "linear"), ("linear", "linear"), ("log", "linear"), ("linear", "linear"), ("log", "linear")),
        (("log", "linear"), ("linear", "linear"), ("log", "symlog"), ("linear", "linear"), ("log", "linear")),
        (("log", "linear"), ("linear", "linear"), ("log", "log"), ("log", "linear"), ("log", "linear"))
    )

    x_labels = ("b", "s1", "t1", "s2", "t2")
    y_labels = ("var($B_{d1}$)", "covar($B_{d1}$,$S_{d1}$)", "var($S_{d1}$)")

    labels = (
        (("lin"), ("lin"), ("log"), ("lin"), ("log")),
        (("log"), ("lin"), ("pow"), ("lin"), ("log")),
        (("con"), ("lin"), ("pow"), ("con"), ("con")),
    )

    for row in range(len(axes)):
        for col in range(len(axes[0])):
            x = xs[col]
            y = v_mats[col][:,:,:,:,:,0,v_indices[row][0],v_indices[row][1]].flatten()
            int_func = int_funcs[row][col]
            axes[row,col].plot(x, y, label="MC")
            y2 = int_func(x[0], x[-1], y[0], y[-1], x)
            axes[row,col].plot(x, y2, label=labels[row][col])
            
            axes[row,col].legend()
            
            axes[row,col].set_xscale(axis_scales[row][col][0])
            axes[row,col].set_yscale(axis_scales[row][col][1])
            
            if row==1 and col==2:
                axes[row,col].set_ylim(1.1*np.amin(y), 0.9*np.amax(y))
                
            if col==0:
                axes[row,col].set_ylabel(y_labels[row])
                
            if row==2:
                axes[row,col].set_xlabel(x_labels[col])
                
    plt.savefig("./main_files/ppc_test/PPC_scaling_different_1_bright_short.pdf", bbox_inches='tight')
    
def scaling_beviour_different_sources_2():
    v_b = calc_bmaxL_variance_matrix(
        np.array(input_b),
        np.array([input_s1[2]]),
        np.array([input_t1[18]]),
        np.array([input_s2[1]]),
        np.array([input_t2[19]])
    )

    v_s1 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array(input_s1),
        np.array([input_t1[18]]),
        np.array([input_s2[1]]),
        np.array([input_t2[19]])
    )

    v_t1 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array([input_s1[2]]),
        np.array(input_t1),
        np.array([input_s2[1]]),
        np.array([input_t2[19]])
    )

    v_s2 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array([input_s1[2]]),
        np.array([input_t1[18]]),
        np.array(input_s2),
        np.array([input_t2[19]])
    )

    v_t2 = calc_bmaxL_variance_matrix(
        np.array([input_b[1]]),
        np.array([input_s1[2]]),
        np.array([input_t1[18]]),
        np.array([input_s2[1]]),
        np.array(input_t2)
)
    
    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12,7))
    fig.tight_layout()

    v_mats = (v_b, v_s1, v_t1, v_s2, v_t2)
    xs = (input_b, input_s1, input_t1, input_s2, input_t2)
    v_indices = ((0,0), (0,1), (1,1))

    b_int_funcs = (interpolate_linear, interpolate_linear, interpolate_logarithmic, interpolate_linear, interpolate_logarithmic)
    c_int_funcs = (interpolate_logarithmic, interpolate_linear, interpolate_powerlaw, interpolate_linear, interpolate_logarithmic)
    s_int_funcs = (interpolate_constant, interpolate_linear, interpolate_powerlaw, interpolate_constant, interpolate_constant)

    int_funcs = (b_int_funcs, c_int_funcs, s_int_funcs)

    axis_scales = (
        (("linear", "linear"), ("linear", "linear"), ("log", "linear"), ("linear", "linear"), ("log", "linear")),
        (("log", "linear"), ("linear", "linear"), ("log", "symlog"), ("linear", "linear"), ("log", "linear")),
        (("log", "linear"), ("linear", "linear"), ("log", "log"), ("log", "linear"), ("log", "linear"))
    )

    x_labels = ("b", "s1", "t1", "s2", "t2")
    y_labels = ("var($B_{d1}$)", "covar($B_{d1}$,$S_{d1}$)", "var($S_{d1}$)")

    labels = (
        (("lin"), ("lin"), ("log"), ("lin"), ("log")),
        (("log"), ("lin"), ("pow"), ("lin"), ("log")),
        (("con"), ("lin"), ("pow"), ("con"), ("con")),
    )

    for row in range(len(axes)):
        for col in range(len(axes[0])):
            x = xs[col]
            y = v_mats[col][:,:,:,:,:,0,v_indices[row][0],v_indices[row][1]].flatten()
            int_func = int_funcs[row][col]
            axes[row,col].plot(x, y, label="MC")
            y2 = int_func(x[0], x[-1], y[0], y[-1], x)
            axes[row,col].plot(x, y2, label=labels[row][col])
            
            axes[row,col].legend()
            
            axes[row,col].set_xscale(axis_scales[row][col][0])
            axes[row,col].set_yscale(axis_scales[row][col][1])
            
            if row==1 and col==2:
                axes[row,col].set_ylim(1.1*np.amin(y), 0.9*np.amax(y))
                
            if col==0:
                axes[row,col].set_ylabel(y_labels[row])
                
            if row==2:
                axes[row,col].set_xlabel(x_labels[col])
                
    plt.savefig("./main_files/ppc_test/PPC_scaling_different_2.pdf", bbox_inches='tight')
    
scaling_beviour_different_sources_1_bright_short()
scaling_beviour_different_sources_2()