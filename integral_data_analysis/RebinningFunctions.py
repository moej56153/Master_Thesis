import numpy as np

def rebin_data_exp(
    bins,
    counts,
    energy_range
):

    if energy_range[0]:
        for i, e in enumerate(bins):
            if e > energy_range[0]:
                bins = bins[i:]
                counts = counts[:,i:]
                break
    if energy_range[1]:
        for i, e in enumerate(bins):
            if e > energy_range[1]:
                bins = bins[:i]
                counts = counts[:,:i-1]
                assert i > 1, "Max Energy is too low"
                break
        
    min_counts = 5
    
    max_num_bins = 120
    min_num_bins = 1
    
    finished = False
    
    while not finished:
        num_bins = round((max_num_bins + min_num_bins) / 2)
        
        if num_bins == max_num_bins or num_bins == min_num_bins:
            num_bins = min_num_bins
            finished = True
        
        temp_bins = np.geomspace(bins[0], bins[-1], num_bins+1)
        
        new_bins, new_counts = rebin_closest(bins, counts, temp_bins)
        
        if np.amin(new_counts) < min_counts:
            max_num_bins = num_bins
        else:
            min_num_bins = num_bins
            
    return new_bins, new_counts

def rebin_data_exp_10(
    bins,
    counts,
    energy_range
):

    if energy_range[0]:
        for i, e in enumerate(bins):
            if e > energy_range[0]:
                bins = bins[i:]
                counts = counts[:,i:]
                break
    if energy_range[1]:
        for i, e in enumerate(bins):
            if e > energy_range[1]:
                bins = bins[:i]
                counts = counts[:,:i-1]
                assert i > 1, "Max Energy is too low"
                break
        
    min_counts = 10
    
    max_num_bins = 120
    min_num_bins = 1
    
    finished = False
    
    while not finished:
        num_bins = round((max_num_bins + min_num_bins) / 2)
        
        if num_bins == max_num_bins or num_bins == min_num_bins:
            num_bins = min_num_bins
            finished = True
        
        temp_bins = np.geomspace(bins[0], bins[-1], num_bins+1)
        
        new_bins, new_counts = rebin_closest(bins, counts, temp_bins)
        
        if np.amin(new_counts) < min_counts:
            max_num_bins = num_bins
        else:
            min_num_bins = num_bins
            
    return new_bins, new_counts
    
def rebin_closest(bins, counts, temp_bins):
    counts = np.copy(counts)
    closest1 = len(bins) - 1
    for i in range(len(temp_bins)-2, 0, -1):
        closest2 = np.argpartition(
            np.absolute(bins - temp_bins[i]),
            0
        )[0]
        if closest1 - closest2 >= 2:
            counts[:,closest2] += np.sum(counts[:, closest2+1 : closest1], axis=1)
            counts = np.delete(
                counts,
                [j for j in range(closest2+1, closest1)],
                axis=1
            )
            bins = np.delete(
                bins,
                [j for j in range(closest2+1, closest1)]
            )
        closest1 = closest2
    return bins, counts