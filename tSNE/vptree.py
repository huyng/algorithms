from collections import namedtuple
import random
import numpy as np
import deque
from Queue import PriorityQueue

### Data structures

Point = namedtuple("Point", "x y")

class VPTree(object):
    def __init__(self, points, dist_fn=l2):
        self.left = None
        self.right = None
        self.radius = None
        self.vp = points.pop(random.randrange(len(points)))
        self.points = points
        if len(points) <= 1:
            return

        # child with all points inside radius
        left_points = []

        # child with all points outside radius    
        right_points = []

        # compute distances
        distances = [dist_fn(self.vp, p) for p in points]

        # choose divisino boundary at median of distances
        self.radius = np.median(distances)

        for i, p in enumerate(points):
            d = distances[i]
            if d >= self.radius:
                right_points.append(p)
            else:
                left_points.append(p)

        if len(left_points) > 0:
            self.left = VPTree(points=left_points, dist_fn=dist_fn)

        if len(right_points) > 0:
            self.right = VPTree(points=right_points, dist_fn=dist_fn)


### Distance functions

def l2(p1, p2):
    return np.sqrt((p2.x-p1.x)**2 + (p2.y-p1.y)**2)

### Operations

def NNSearch(tree, query, n, dist_fn=l2):
    if tree is None:
        return
    neighbors = PriorityQueue(n)
    visit_stack = deque([tree])
    tau = np.inf
    while len(visit_stack) > 0:
        node = visit_stack.pop()
        d = dist(query, node.vp)






if __name__ == '__main__':
    X = np.random.uniform(0, 1000, size=40000)
    Y = np.random.uniform(0, 1000, size=40000)
    points = [Point(x,y) for i, (x, y) in  enumerate(zip(X,Y))]
    tree = VPTree(points)

            