
import collections

import scipy

from interface import Function

# Operator functions for combining things.

class Operator ( Function ) :

    def __init__( self, function_1, function_2 ) :
        self.function_1 = function_1
        self.function_2 = function_2

    @property
    def hyperparameters( self ) :
        deque = self.function_1.hyperparameters
        deque.extend( self.function_2.hyperparameters )
        return deque

    def take_hyperparameters( self, deque ) :
        self.function_1.take_hyperparameters( deque )
        self.function_2.take_hyperparameters( deque )

class Sum ( Operator ) :

    def __call__( self, inputs ) :
        return self.function_1( inputs ) + self.function_2( inputs )

class Product ( Operator ) :

    def __call__( self, inputs ) :
        return self.function_1( inputs ) * self.function_2( inputs )

# Mean functions.

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

# Covariance functions.

class CovarianceFunction ( Function ) :
    pass

class GaussianWhiteNoiseCovariance ( CovarianceFunction ) :

    def __init__( self, amplitude ) :
        self.amplitude = amplitude

    def __call__( self, input_diffs ) :
        return self.amplitude ** 2 * scipy.eye( len( input_diffs ) )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.amplitude ] )

    def take_hyperparameters( self, deque ) :
        self.amplitude = deque.popleft()

class IsotropicSquaredExponentialCovariance ( CovarianceFunction ) :

    def __init__( self, amplitude, length_scale ) :
        self.amplitude    = amplitude
        self.length_scale = length_scale

    def __call__( self, input_diffs ) :
        scaled_diffs = input_diffs / self.length_scale
        return self.amplitude ** 2 * scipy.exp( - scipy.sum( scaled_diffs ** 2, 2 ) )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.amplitude, self.length_scale ] )

    def take_hyperparameters( self, deque ) :
        self.amplitude    = deque.popleft()
        self.length_scale = deque.popleft()

class AnisotropicSquaredExponentialCovariance ( CovarianceFunction ) :

    def __init__( self, amplitude, length_scales ) :
        self.amplitude     = amplitude
        self.length_scales = length_scales

    def __call__( self, input_diffs ) :
        scaled_diffs = input_diffs / self.length_scales
        return self.amplitude ** 2 * scipy.exp( - scipy.sum( scaled_diffs ** 2, 2 ) )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.amplitude ] + self.length_scales.tolist )

    def take_hyperparameters( self, deque ) :
        self.amplitude = deque.popleft()
        for i in range( len( self.length_scales ) ) :
            self.length_scales[ i ] = deque.popleft()

class IsotropicMaternCovariance_3_2 ( CovarianceFunction ) :

    def __init__( self, amplitude, length_scale ) :
        self.amplitude    = amplitude
        self.length_scale = length_scale

    def __call__( self, input_diffs ) :
        scaled_diffs = input_diffs / self.length_scale
        arg = scipy.sqrt( 3.0 * scipy.sum( scaled_diffs ** 2, 2 ) )
        return self.amplitude ** 2 * ( 1.0 + arg ) * scipy.exp( - arg )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.amplitude, self.length_scale ] )

    def take_hyperparameters( self, deque ) :
        self.amplitude    = deque.popleft()
        self.length_scale = deque.popleft()

class IsotropicMaternCovariance_5_2 ( CovarianceFunction ) :

    def __init__( self, amplitude, length_scale ) :
        self.amplitude    = amplitude
        self.length_scale = length_scale

    def __call__( self, input_diffs ) :
        scaled_diffs = input_diffs / self.length_scale
        arg1 = 5.0 * scipy.sum( scaled_diffs ** 2, 2 )
        arg2 = scipy.sqrt( arg1 )
        arg1 /= 3.0
        return self.amplitude ** 2 * ( 1.0 + arg1 + arg2 ) * scipy.exp( - arg2 )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.amplitude, self.length_scale ] )

    def take_hyperparameters( self, deque ) :
        self.amplitude    = deque.popleft()
        self.length_scale = deque.popleft()
