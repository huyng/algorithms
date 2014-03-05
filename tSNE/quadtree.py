MAX_POINTS_PER_NODE = 4
MAX_DEPTH = 6
NW = 0
NE = 1
SE = 2
SW = 3

class Rectangle(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.x_max = x + width
        self.y_max = y + height
        self.width = width
        self.height = height

    @property
    def midpoint(self):
        x_mid = (self.x + self.x_max) * 0.5
        y_mid = (self.y + self.y_max) * 0.5
        return Point(x_mid, y_mid)

    def contains(self, point):
        return (self.x <= point.x < self.x_max) and (self.y <= point.y < self.y_max)

    def __repr__(self):
        return "{xmin: %-8s, ymin: %-8s, xmax: %-8s, ymax: %-8s}" % (self.x, self.x_max,  self.y, self.y_max)

class Point(object):
    def __init__(self, x, y, idx=None):
        self.x = x
        self.y = y
        self.idx = idx  # used to keep track of the original data structure
        
class QuadTree(object):
    def __init__(self, x_min, x_max, y_min, y_max, depth=0):
        self._depth = depth
        self.bounds = Rectangle(x_min, y_min, x_max-x_min, y_max-y_min)
        self.quads = None
        self.points = []
        self._has_subdivided = False
        self.cmass = None  # center of mass
        self.size = 0  # toal number of points contained in node

    def subdivide(self):
        depth = self._depth + 1
        self.quads = [None, None, None, None]
        b = self.bounds
        m = b.midpoint
        self.quads[NW] = QuadTree(b.x, m.x, b.y, m.y, depth=depth)
        self.quads[NE] = QuadTree(m.x, b.x_max, b.y, m.y, depth=depth)
        self.quads[SE] = QuadTree(m.x, b.x_max, m.y, b.y_max, depth=depth)
        self.quads[SW] = QuadTree(b.x, m.x, m.y, b.y_max, depth=depth)
        for point in self.points:
            self._insert_into_sub_quad(point)
        self.points = None
        self._has_subdivided = True

    def _insert_into_sub_quad(self, point):
        for quad in self.quads:
            if quad.bounds.contains(point):
                quad.insert(point)
                break

    def insert(self, point):
        # online update of center of mass
        self.size += 1
        if self.cmass is None:
            self.cmass = point
        else:
            x = ((self.size-1)*self.cmass.x+point.x)/self.size
            y = ((self.size-1)*self.cmass.y+point.y)/self.size
            self.cmass = Point(x,y)


        if self._has_subdivided:
            self._insert_into_sub_quad(point)
        else:
            self.points.append(point)
            if len(self.points) >= MAX_POINTS_PER_NODE and self._depth < MAX_DEPTH:
                self.subdivide()


# https://stackoverflow.com/questions/5278580/non-recursive-depth-first-search-algorithm
def traverse(tree, visit, mode="bfs"):
    from collections import deque
    visit_stack = deque([tree])
    pop = visit_stack.popleft if mode=="bfs" else visit_stack.pop
    while len(visit_stack) > 0:
        node = pop()
        if node.quads is not None:
            visit_stack.extend(node.quads)
        visit(node)


if __name__ == '__main__':
    import numpy as np
    from matplotlib.patches import Rectangle as MPLRect
    from matplotlib.patches import Circle
    from matplotlib import pyplot as plt

    X = np.random.uniform(0, 100, size=40)
    Y = np.random.uniform(0, 100, size=40)
    points = [Point(x,y, idx=i) for i, (x, y) in  enumerate(zip(X,Y))]

    qt = QuadTree(0, 100, 0, 100)
    for p in points:
        qt.insert(p)

    ax = plt.gca()
    def visit(node):
        # print node.bounds, "->", node._depth
        # cell = MPLRect((node.bounds.x, node.bounds.y), 
        #                 node.bounds.width, 
        #                 node.bounds.height,
        #                 fill=False,
        #                 edgecolor="blue")
        # ax.add_patch(cell)
        # if node.cmass:
        #     mass = Circle((node.cmass.x, node.cmass.y), radius=.5, fc="red", fill=True)
        #     ax.add_patch(mass)
        pass


    traverse(qt, visit, "dfs")
    plt.scatter(X,Y, s=8)
    plt.show()

