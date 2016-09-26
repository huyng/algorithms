This is a distributed connected components (union-find) implementation for
pyspark based on the following paper.

    http://www.cse.unr.edu/~hkardes/pdfs/ccf.pdf

Usage:

    from pyspark import SparkContext, SparkConf
    import ccf_spark as ccf

    conf = SparkConf()
    sc = SparkContext(conf)

    edges = sc.parallelize([
        ("a", "b"),
        ("b", "c"),
        ("d", "e"),
        ("d", "f"),
        ("g", "b")
    ])

    vertices_to_roots = ccf.ccf_run(sc, edges, max_iters=5)
    root_to_children = ccf.ccf_group_by_root(vertices_to_roots)
    print root_to_children.take(10)

Expected output:

    [
        ("a", ["a", "b", "c", "g"]),
        ("d", ["e", "f"])
    ]

    
    

