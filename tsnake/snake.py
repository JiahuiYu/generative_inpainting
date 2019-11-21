"""
Module containing the implementation of parts of the paper
up to and including Section 3.2 of the paper. 
"""

import numpy as np

# Implementation Notes: https://www.crisluengo.net/archives/217#more-217

class Node(object):
    """
    A class representing a single node in a T-snake.
    """
    
    def __init__(self, x, y):
        self.x = x  # todo: arg check if we care
        self.y = y
       
    @property
    def position(self):
        return np.array([x, y]).reshape(1, 2)
        
# @allen: Not sure if we'll need this element, but leaving it for now        
class Element(object):
    """
    Class representing an element / edge between two nodes in the T-snake.
    
    The Tsnake class instantiates the Elements automatically in its constructor, 
     so directly calling this constructor elsewhere (probably) shouldn't be necessary.
     (NOTE: can change if inconvenient)
    """
    
    def __init__(self, node1, node2):
        assert isinstance(node1, Node)
        assert isinstance(node2, Node)
        self.node1 = node1
        self.node2 = node2
        
    def intersects_grid_cell(self, grid_cell):
        # TODO: clean this up
        raise NotImplementedError


class TSnake(object):
    """
    A class representing a *single* T-snake at a given instance in time.
    
    If the initial T-snake splits into two, then there should be *two*
      instances of T-snakes present in the algorithm.
      (NOTE: can change this if inconvenient)
      
    The merging two T-snakes in the algorithm should be done with:
      TSnake.merge(snake1, snake2)
      
    In this class, 
        each element/edge       self.elements[i] 
            corresponds to:
        node pair               (self.nodes[i], self.nodes[i+1])
    
    Args:
    ===========================================
    (list) nodes:
    * A list of Node instances, in order i=0, 1, 2, ..., N-1
    ===========================================
    
    TODO: hyperparameters in constructor
    """
    
    def __init__(self, nodes, force, intensity):
        for n in nodes:
            assert isinstance(n, Node)
        self.nodes = list(nodes)
        # Force and intensity fields over the image, (n,m) np arrays
        self.force = force
        self.intensity = intensity
        
        self.elements = []
        # Connect each node[i] --> node[i+1]
        for i in range(len(nodes)-1):
            self.elements.append(Element(self.nodes[i], self.nodes[i+1]))
            
        # Connect node[N-1] --> node[0]
        self.elements.append(Element(self.nodes[-1], self.nodes[0]))
    
    @property
    def num_nodes(self):
        return len(self.nodes)
    
    @property
    def node_locations(self):
        """
        # TODO: Store or compute normals somewhere
        Returns an (N, 2) matrix containing the current node locations for this T-snake.
        ( In the paper: $\bm{x}(t)$ )
        """
        N = self.num_nodes
        locs = [node.position for node in self.nodes]
        return np.array(locs).reshape(N, 2)
    
    def compute_alpha(self):
        """ Eq 2 """
        raise NotImplementedError
        
    def compute_beta(self):
        """ Eq 3 """
        raise NotImplementedError
        
    def compute_rho(self):
        """ Eq 4 """
        raise NotImplementedError
        
    def compute_f(self):
        """ Eq 7 """
        raise NotImplementedError
    
    def compute_potential(self):
        """
        P(
        """
        raise NotImplementedError
    
    def step(self, dt):
        """
        Args:
        ===========================================
        (float) dt:
        * Non-negative number representing step-size / time-step.
        ===========================================
        """
        raise NotImplementedError
    
    @classmethod
    def merge(cls, snake1, snake2):
        """
        Merge two (previously-split) T-snakes together.
        """
        raise NotImplementedError
        
    def update_snake_nodes(self, new_nodes):
        """
        Updates snake with new nodes defined in np array of length n
        new_nodes of Node() objects
        """
        raise NotImplementedError
    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
if __name__ == '__main__':
    pass






