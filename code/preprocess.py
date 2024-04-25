import numpy as np

# converts accel values to richter magnitudes
def accel_to_rich(accel):
    g = accel / 980.665
    mercalli_split = [.000464, .00175, .00297, .0276, .062, .115, .215, .401, .747, 1.39]
    ratios = [val / next((mval for mval in mercalli_split if val < mval), mercalli_split[-1]) for val in g]
    mercalli_ids = np.digitize(g, mercalli_split) + 1
    mercalli_richter = {1:1, 2:3, 3:3.5, 4:4, 5:4.5, 6:5, 7:5.5, 8:6, 9:6.5, 10:7, 11:7.5, 12:8}
    richter_vals = [mercalli_richter[id] for id in mercalli_ids] 
    richter_vals += ratios
    # print(richter_vals)
    return richter_vals