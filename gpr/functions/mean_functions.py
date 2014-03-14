
# Mean functions.

import collections

import scipy

from ..interface import Function

class MeanFunction ( Function ) :
    pass

class ZeroMean ( MeanFunction ) :

    def __call__( self, inputs ) :
        return scipy.zeros( len( inputs ) )

    @property
    def hyperparameters( self ) :
        return collections.deque( [] )

    def take_hyperparameters( self, deque ) :
        pass

class ConstantMean ( MeanFunction ) :

    def __init__( self, constant ) :
        self.constant = constant

    def __call__( self, inputs ) :
        return self.constant * scipy.ones( len( inputs ) )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.constant ] )

    def take_hyperparameters( self, deque ) :
        self.constant = deque.popleft()

class PolynomialMean ( MeanFunction ) :

    def __init__( self, coefficients ) :
        self.coefficients = coefficients

    def __call__( self, inputs ) :
        mean_input = inputs.mean()
        return scipy.polyval( self.coefficients, inputs - mean_input ).flatten()

    @property
    def hyperparameters( self ) :
        return collections.deque( self.coefficients )

    def take_hyperparameters( self, deque ) :
        for i in range( len( self.coefficients ) ) :
            self.coefficients[ i ] = deque.popleft()
