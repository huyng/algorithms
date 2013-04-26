
def recursive_avg(y, previous_avg, total):
    """
    computes mean recursively

    m = 0
    for j,i in enumerate(range(0,10)):
        m = recursive_mean(i, m, j+1)
        print m

    """
    y = float(y)
    return (previous_avg * (total-1) + y)/total

def ewma(y, previous_avg, weight=0.1):
    """
    exponential weighted moving average
    m = 0
    for i in range(1000):
        m = ewma(i,m, .02)
        print m
    """
    return float(weight) * y + (1 - weight) * previous_avg
