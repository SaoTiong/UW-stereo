import numpy as np
d = np.load('/home/tong/recordings/uwstereo_test/tower/pred2/disparity/008715.npy')
print(d.shape, d.dtype, d.min(), d.max(), np.isnan(d).any())
