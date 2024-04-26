import numpy as np

def accel_to_g(accel):
    return np.array(accel) / 980.665

# converts accel values to richter magnitudes
def accel_to_rich(accel):
    g = accel_to_g(accel)
    mercalli_split = [.000464, .00175, .00297, .0276, .062, .115, .215, .401, .747, 1.39]
    ratios = [val / next((mval for mval in mercalli_split if val < mval), mercalli_split[-1]) for val in g]
    mercalli_ids = np.digitize(g, mercalli_split) + 1
    mercalli_richter = {1:1, 2:3, 3:3.5, 4:4, 5:4.5, 6:5, 7:5.5, 8:6, 9:6.5, 10:7, 11:7.5, 12:8}
    richter_vals = np.array([mercalli_richter[id] for id in mercalli_ids]) + ratios
    return richter_vals