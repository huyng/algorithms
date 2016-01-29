"""
This a distributed connected components implementation for
spark based on the following paper.

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
    root_to_children.take(10)

"""

def ccf_iterate(sc, edges):

    found_new_pair = sc.accumulator(0)

    def cc_mapper(edge):
        """
        edge here is a tuple of vertices (v_a, v_b).

        We simply emit an edge from v_a to v_b and from v_b to v_a
        """
        v_a, v_b = edge
        return [
            (v_a, v_b),
            (v_b, v_a)
        ]

    def ccf_find_root(group):
        """
        group is a tuple of (current_vertex_id, [neighbor_vertex_1, ... neighbor_vertex_N])

        We consider the smallest vertex_id in this group to be the "root", and
        emit an edge for every vertex in this group to its root:
        [
            (vertex_id_1, root_vertex_id),
            (vertex_id_2, root_vertex_id),
            ...
        ]
        """
        key, values = group

        values = list(values)
        smallest = min(values + [key])
        emit = []

        # check to see if there are any vertices smaller than current_vertex_id,
        # if not, the current vertex is the root and all other nodes should point to it
        if smallest < key:
            emit.append((key, smallest))
            for v in values:
                if v != smallest:
                    emit.append((v, smallest))
                    found_new_pair.add(1)
        return emit

    return edges.flatMap(cc_mapper).groupByKey().flatMap(ccf_find_root), found_new_pair


def ccf_run(sc, edges, max_iters=5):
    """
    This function takes an RDD of edges in the form of (vertex_id1, vertex_id2)
    and produces an RDD of vertices and their root's id (vertex_id, root_vertex_id)
    """

    while True:
        edges.persist()
        new_edges, found_new_pair = ccf_iterate(sc, edges)
        new_edges.take(1)
        max_iters -= 1

        print("-- remaing steps: %d found_new_pair: %d" % (max_iters, found_new_pair.value))

        if max_iters <= 0:
            return edges
        if found_new_pair.value == 0:
            return edges

        edges.unpersist()
        edges = new_edges


def ccf_group_by_root(vertex_to_root):
    """
    Assuming vertex_to_root is an RDD of (vertex_id, root_vertex_id),
    this function produces an RDD of the form:

        (root_vertex_id, [child_vertex_id1 ... child_vertex_idN])
    """
    return vertex_to_root.new_edges.map(lambda x: (x[1], x[0]))\
                         .groupByKey()\
                         .mapValues(list)
