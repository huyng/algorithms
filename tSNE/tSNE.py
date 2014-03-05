import numpy as np

def l2_sqr(x1, x2):
    """
    computes euclidean distance squared between 2 points
    """
    norm_squared = np.sum(np.square(x1 - x2))
    return norm_squared

def scale_features(X):
    """
    center features around zero mean and scale features
    """
    m, n = X.shape
    s = X.sum(axis=0)
    mu = s/float(m)
    Y = X - mu
    ymax = np.max(Y, axis=0)
    ymin = np.min(Y, axis=0)
    ydiff = ymax - ymin
    Y = Y/ydiff
    return Y

class TSNE(object):
    """
    This is the tSNE model
    """
    def __init__(self, perplexity=5):
        """
        perplexity
        """
        self.perplexity = perplexity
        pass

    def fit(self, X):
        # p_ji a.k.a the pairwise affinity
        def similarity(x1, x2, d, s): 
            return np.exp(-1.0*d(x1,x2)**2/(2*s**2)
        def d(x1,x2):
            return x1-x2
        for i in range(X.shape[0])
            xi = X[i]
            for j in range(X.shape[0]):
                xj = X[j]
                dist = lambda x1,x2: 
                numer = similarity(xi, xj, )
                denom = 0
                for k in range(X.shape[0]):
                    if k == i:
                        continue
                    xk = X[k]
                    denom +=  np.exp(-1.0 * l2_sqr(xi, xk)/sigma)
            p_ji = numer/denom


    

if __name__ == '__main__':
    model = TSNE()
    model.fit()
    model.transform()




        