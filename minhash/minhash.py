"""

We need a way to describe sets A, B, C, D ... 
such that if the description of any two sets
are equal, we can know how similar the sets are 
to each other.

                             |A ∪ B|
P(min(H(A)) = min(H(B)) ) = -------
                             |A ∩ B|


- Let x be a member in (A union B)
- What's the probability that x is in both A and B?
- Turns out that this probablity is equal to the jaccard coeeff which can be used to measure
  similarity between two sets.
- Let H() be a hashing function
- H(A) = [ h(x) for x in A ]
- min(H(A)) = min(H(B)) if and only if the x producing the minimum hash is in A and B
- the probability that x is in both A and B is the jaccard coeff
- we compute many minhashes for a set
- to estimate similaryt of sets, we can:
    1) summarize sets as vectors of minhashes
    2) test for equality of each vector component at query time to get similarity

"""

import hashlib

def default_hash(i):
    return hashlib.md5(i).digest()[:8]  # return 8 bytes (64 bits)

def summary_vector(A, k=10, hash_fn=default_hash):
    """
    Compute the minhash summary vector for 
    a given set.

    :arg A:          The set
    :arg k:          The # of hash outputs to be used to estimate the 
    :arg hash_fn:    The hashing function to use
    """

    # hash each element in set A
    A_hashed = [hash_fn(i) for i in A]

    # summarize A as the k minimum hashes
    A_summary = sorted(A_hashed)[:k]

    return A_summary

def estimate_similarity(A, B):
    """
    Given two sets, use minhash summary vectors to 
    estimate similarity as defined by jaccard coefficient
    """

    SA = summary_vector(A)    
    SB = summary_vector(B)
    min_matches = sum([hminA == hminB for hminA , hminB in zip(SA, SB)])
    compare_len = min([len(SA), len(SB)])
    estimate = float(min_matches)/float(compare_len)
    return estimate
