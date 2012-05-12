import numpy as np

def tfidf_weights(term, documents):
    """
    Calculate the term-frequency inverse-document-frequency
    for a given term in a set of documents

    :arg term:          A term
    :arg documents:     A list of documens composed of terms

    >>> docs = [(1,2,5,1,3,2,1), (1,3,14,5,1,31,2,5), (102,2,3,56,7)]
    >>> term = 1
    >>> print tfidf_weights(term, docs)

    """
    tf  = [d.count(term) for d in documents]  # term frequency in each documents
    docs_containing_term = sum(term in d for d in documents) or 1
    docs_count = len(documents)  # document cardinality
    idf = np.log(float(docs_count) / float(docs_containing_term))
    return np.array(tf) * idf
