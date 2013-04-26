"""
Term Frequency vs Inverse Document Frequency (tf-idf)
is a common measure used for ranking how likely a
document matches a search term.
"""

import numpy as np

def tfidf_weights(term, documents):
    """
    Calculate the term-frequency inverse-document-frequency
    for a given term in a set of documents.

    :arg term:          A term
    :arg documents:     A list of documens composed of terms

    >>> docs = [(1,2,5,1,3,2,1), (1,3,14,5,1,31,2,5), (102,2,3,56,7)]
    >>> term = 1
    >>> print tfidf_weights(term, docs)

    """

    # term frequency in each documents
    tf = [d.count(term)/float(len(d)) for d in documents]  

    # document cardinality
    docs_count = len(documents) 
    docs_containing_term = sum(term in d for d in documents) or 1
    idf = np.log(float(docs_count) / float(docs_containing_term))
    return np.array(tf) * idf
