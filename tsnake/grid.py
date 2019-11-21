"""
Module containing implementations of the ACID technique.
"""

import numpy as np
import snake as snake

class GridCellEdge(object):
    """
    Represents one of the sides / cell-edges in the grid. 
    """
    
    def __init__(self, coord1, coord2):
        """
        Represents one grid cell edge (one of three components of a TriangeCell).
        
        Args:
        ==========================================
        (2-tuple) coord1, coord2:
        * The (x, y) points that form this line segment.
        * The top-left corner of the image-rectangle should be (0, 0).
          (to be compatible with numpy indexing) (can change this if inconvenient)
        ==========================================
        """
        self._coord1 = coord1  # todo maybe argchecks 
        self._coord2 = coord2
        
        # TODO: implement this data structure
        self.intersections = dict()
        
    @property
    def endpoints(self):
        return {self._coord1, self._coord2}

    def find_intersection_point_with_element(self, element):
        """
        If this grid cell edge intersects with the given element:
            - Return the point of intersection (2-tuple of the coordinates).
        Else:
            - Return None
        """
        raise NotImplementedError
    

class CellGrid(object):
    """
    Class representing the entire cell grid (of triangles) for an image.
      - image meaning the blank space the T-snake is segmenting / infilling
      - assumes that each triangle-cell is a right triangle
        (for the Coxeter-Freudenthal triangulation) (see Fig 2 in the paper)
        
      - assumes (for now) that the space we're segmenting / infilling is rectangular
    
    
    In the paper, Demetri mentions the 'Freudenthal triangulation' for 
    implementing the cell-grid:
     https://www.cs.bgu.ac.il/~projects/projects/carmelie/html/triang/fred_T.htm
    
    Args:
    ==========================================
    (np.array) image:
    * (n by m) matrix representing the gray-scale image.

    (float) scale:
    * float between 0 and 1 representing the number of pixels per cell, i.e. 1=1 vertex/pixel, .5 = 2 vertex per pixel, so on
    
    ==========================================
    """
    
    def __init__(self, image, scale=1.0):
        """
        @allen: Should we pass a snake to the board? should the board own the snake?
        TODO: implement Freudenthal triangulation
        https://www.cs.bgu.ac.il/~projects/projects/carmelie/html/triang/fred_T.htm
        """
        assert isinstance(image, np.array)
        assert len(image.shape) == 2
        assert type(scale) == float
        self.image = image
        self.m, self.n = image.shape
        self.scale = scale
        self.n  = None # TODO: Implement gen_simplex_grid
        self.edges = dict() # Hash set containing [sorted((x1, y1), (x2,y2))]:edge, sort to resolve ambiguity
        self.force_grid = None # same size as grid
        self.intensity_grid = None # same size as grid
        self.snakes = list() # All the snakes on this grid
    
        raise NotImplementedError

    def gen_simplex_grid(self):
        """
        Private method to generate simplex grid and edge map over image at given scale
        self.grid = np array of size (n/scale) * m/scale
        
        * Verticies are on if positive, off if negative, and contain 
            bilinearly interpolated greyscale values according to surrouding pixels
        
        * vertex position indicated by its x and y indicies
        """   
        # for point in grid:
        #     for neighbor points:
        #         if sorted(point, neighbor) not in self.edges:
        #             add it to self.edges

        raise NotImplementedError

    def gen_force_grid(self):
        """
        generate force grid over simplex grid, will need to interpolate
        """
        raise NotImplementedError

    def gen_intensity_grid(self):
        """
        generate intensity grid over simpelx grid, will need to interpolate
        """
        raise NotImplementedError

    def add_snake(self, new_snake):
        """
        Add a new snake to the grid
        """
        assert isinstance(new_snake, snake.Snake)
        self.snakes.append(new_snake)
    
    def compute_intersections(self, snake):
        """
        Compute intersections between the grid and the snake in question
        """
        # Get snake nodes, see which grid verticies they're closest to,
        # grab all the edges between the verticies they're closest to, 
        # compute intersections
        # enqueue grid vertex inside the snake

    
    
    
    
    
    
    
    
### TODOS ###
# 1. snake updates - Joe
# 2. gan mask -> snake -> gan mask - Cole
# 3. algo phase 1: grid intersections - Allen
# 4. algo phase 2: turning nodes on / off, remove inactive points - Eric
    
    
    
    
    

    
    
if __name__ == '__main__':
    pass









