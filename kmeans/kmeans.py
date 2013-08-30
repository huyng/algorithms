import random
import numpy

def KmeansCluster(X, k, t=.0000001, max_iter=100):
    """
    kmeans written c-style
    """
    # random initialization of centroids
    n = len(X[0])
    m = len(X)
    xmax = X.max()
    xmin = X.min()
    centroids = [[random.uniform(xmin, xmax) for j in range(n)] for i in range(k)]
    labels = [-1 for i in range(m)]
    old_error = 99999999
    error = 0
    iteration = 0

    while 1:
        error = 0

        # assign
        counts = [0 for j in range(k)]
        accumulator = [[0 for l in range(n)] for j in range(k)]
        for i in xrange(m):
            min_d = 99999999
            for j in xrange(k):
                d = 0
                for l in xrange(n):
                    d += abs(centroids[j][l] - X[i][l])
                if d < min_d:
                    labels[i] = j
                    min_d = d
            counts[labels[i]] += 1
            error += min_d
            for l in range(n):
                accumulator[labels[i]][l] += X[i][l]

        # update
        for j in range(k):
            for l in range(n):
                accumulator[j][l] /= float(counts[j]+1)
        centroids = accumulator
        print "iteration ", iteration
        print "error ", error

        max_iter -= 1
        iteration += 1
        if abs(old_error-error) < t or max_iter <= 0:
            break
        old_error = error
    return labels

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    X = numpy.loadtxt("data.csv",delimiter=",")
    idx = KmeansCluster(X, k=3)
    colors = ([([0.4,1,0.4],[1,0.4,0.4],[0.1,0.8,1])[i] for i in idx])
    plt.scatter(X[:,0], X[:,1], c=colors)
    raw_input()
