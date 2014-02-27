MAX_POINTS_PER_NODE = 4

def is_between(x, start, stop):
    return start <= x < stop

def midpoint(start, stop):
    return round((start + stop)*0.5)

class QuadTree(object):
    def __init__(self, x_min, x_max, y_min, y_max, max_depth=4):
        self._max_depth = max_depth
        x_mid = midpoint(x_min, x_max)
        y_mid = midpoint(x_min, x_max)
        self.x_min = x_min
        self.y_min = y_min
        self.y_max = y_max
        self.x_max = x_max
        self.x_mid = midpoint(x_min, x_max)
        self.y_mid = midpoint(y_min, y_max)
        
        # https://stackoverflow.com/questions/20837530/quadtree-nearest-neighbour-algorithm
        # ordered in clockwise fashion, start from nw quadrant
        self.quadrants = {
            (x_min, x_mid, y_min, y_mid): None  # north-west
            (x_mid, x_max, y_min, y_mid): None  # north-east
            (x_mid, x_max, y_mid, y_max): None  # north-east
            (x_min, x_mid, y_mid, y_max): None  # north-east
        }
        self.points = []

    def insert(self, x, y):

        # if there are less than 4 points then add this point
        if len(self.points) < MAX_POINTS_PER_NODE:
            self.points.append(point)
        else:
            self.points

if __name__ == '__main__':
    pass
