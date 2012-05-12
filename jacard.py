def jacard_index(A,B):
    """
    The jacard index gives the similarity between two sets of items:

    :arg A: set A
    :arg B: set B
    :rtype: A 
    
    http://en.wikipedia.org/wiki/Jaccard_index

    """

    A = set(A)
    B = set(B)
    intersection = A.intersection(B)
    union = A.union(B)
    return float(len(intersection))/float(len(union))


def jacard_distance(A, B):
    """
    measures the disimmilarity betwee two sets
    """
    return 1 - jacard_index(A,B)
