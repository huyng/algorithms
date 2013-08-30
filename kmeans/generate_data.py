import numpy as np
pt1 = np.random.normal(1, 0.2, (2000,2))
pt2 = np.random.normal(2, 0.5, (3000,2))
pt3 = np.random.normal(3, 0.3, (1000,2))
pt3[:,0] += 3.5
pt1[:,1] -= 2.5

points = np.concatenate([pt1,pt2,pt3])
points = points.astype(np.float32)
np.savetxt("data.csv", points, delimiter=",", fmt="%.3f")
