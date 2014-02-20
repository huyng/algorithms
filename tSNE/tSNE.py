import numpy as np

def l2_sqr(x1, x2):
    """
    computes euclidean distance squared between 2 points
    """
    norm_squared = np.sum(np.square(x1 - x2))
    return norm_squared


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
        for i in range(X.shape[0])
            xi = X[i]
            sigma = 2*np.square(sigma)
            for j in range(X.shape[0]):
                xj = X[j]
                numer = np.exp(-1.0 * l2_sqr(xi, xj)/sigma)
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




        