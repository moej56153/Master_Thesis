import numpy as np

# def rebin_data_exp(
#     bins,
#     counts,
#     energy_range
# ):

#     if energy_range[0]:
#         for i, e in enumerate(bins):
#             if e >= energy_range[0]:
#                 bins = bins[i:]
#                 counts = counts[:,i:]
#                 break
#     if energy_range[1]:
#         for i, e in enumerate(bins):
#             if e > energy_range[1]:
#                 bins = bins[:i]
#                 counts = counts[:,:i-1]
#                 assert i > 1, "Max Energy is too low"
#                 break
        
#     min_counts = 5
    
#     max_num_bins = 120
#     min_num_bins = 2
    
#     finished = False
    
#     while not finished:
#         num_bins = round((max_num_bins + min_num_bins) / 2)
        
#         if num_bins == max_num_bins or num_bins == min_num_bins:
#             num_bins = min_num_bins
#             finished = True
        
#         temp_bins = np.geomspace(bins[0], bins[-1], num_bins+1)
        
#         new_bins, new_counts = rebin_closest(bins, counts, temp_bins)
        
#         if np.amin(new_counts) < min_counts:
#             max_num_bins = num_bins
#         else:
#             min_num_bins = num_bins
            
#     return new_bins, new_counts

def rebin_data_exp_50(
    bins,
    counts,
    energy_range
):

    if energy_range[0]:
        for i, e in enumerate(bins):
            if e >= energy_range[0]:
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
        
    min_counts = 50
    
    max_num_bins = 120 # these bin numbers do not necessarily correlate to final bin numbers
    min_num_bins = 2
    
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
    for i in range(len(temp_bins)-2, -1, -1):
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


def spimodfit_binning_SE(bins, counts, energy_range):
    
    
    new_bins = np.array([20.0, 21.5, 23.5, 25.5, 27.5, 30.0, 32.5, 35.5, 38.5, 42.0, 45.5, 49.5, 54.0, 58.5, 63.5,
                         69.0, 75.0, 81.5, 89.0, 96.5, 105.0, 114.0, 124.0, 134.5, 146.0, 159.0, 172.5, 187.5, 204.0,
                         221.5, 240.5, 261.5, 284.0, 308.5, 335.5, 364.5, 396.0, 430.0, 467.5, 508.0, 514.0, 600.0,])
    
    new_counts = np.zeros((counts.shape[0], len(new_bins)-1))
    
    for i in range(len(new_bins)-1):
        assert new_bins[i] in bins, f"{new_bins[i]} not in energy_bins"
        assert new_bins[i+1] in bins, f"{new_bins[i+1]} not in energy_bins"
        
        index1 = np.where(bins == new_bins[i])[0][0]
        index2 = np.where(bins == new_bins[i+1])[0][0]
        
        new_counts[:,i] = np.sum(counts[:, index1 : index2], axis=1)
    
    if energy_range[0]:
        for i, e in enumerate(new_bins):
            if e >= energy_range[0]:
                new_bins = new_bins[i:]
                new_counts = new_counts[:,i:]
                break
    if energy_range[1]:
        for i, e in enumerate(new_bins):
            if e > energy_range[1]:
                new_bins = new_bins[:i]
                new_counts = new_counts[:,:i-1]
                assert i > 1, "Max Energy is too low"
                break
    
    return new_bins, new_counts

def exp_binning_function_for_x_number_of_bins(num_bins):
    def binning_function(
        bins,
        counts,
        energy_range
    ):
        if energy_range[0]:
            for i, e in enumerate(bins):
                if e >= energy_range[0]:
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
        
        temp_bins = np.geomspace(bins[0], bins[-1], num_bins+1)
        
        new_bins, new_counts = rebin_closest(bins, counts, temp_bins)
                
        return new_bins, new_counts
    
    return binning_function

def no_rebinning(bins, counts, energy_range):
    if energy_range[0]:
        for i, e in enumerate(bins):
            if e >= energy_range[0]:
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
        
    return bins, counts
        