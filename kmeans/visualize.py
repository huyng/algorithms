import numpy as np
from matplotlib import pyplot as plt
data = np.loadtxt("clusters.csv", delimiter=",")
X = data[:, 1:3]
idx = data[:,0].astype(np.uint32)
cmap = ([0.4,1,0.4], [1,0.4,0.4], [0.1,0.8,1])
colors = [cmap[i] for i in idx]
plt.scatter(X[:,0], X[:,1], c=colors)
raw_input()

