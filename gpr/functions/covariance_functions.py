
# Covariance functions.

import collections

import scipy

from interface import Function

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
        return collections.deque( [ self.amplitude ] + self.length_scales.tolist() )

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

class AnisotropicMaternCovariance_3_2 ( CovarianceFunction ) :

    def __init__( self, amplitude, length_scales ) :
        self.amplitude     = amplitude
        self.length_scales = length_scales

    def __call__( self, input_diffs ) :
        scaled_diffs = input_diffs / self.length_scales
        arg = scipy.sqrt( 3.0 * scipy.sum( scaled_diffs ** 2, 2 ) )
        return self.amplitude ** 2 * ( 1.0 + arg ) * scipy.exp( - arg )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.amplitude ] + self.length_scales.tolist() )

    def take_hyperparameters( self, deque ) :
        self.amplitude = deque.popleft()
        for i in range( len( self.length_scales ) ) :
            self.length_scales[ i ] = deque.popleft()

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

class AnisotropicRationalQuadraticCovariance ( CovarianceFunction ) :

    def __init__( self, amplitude, sqrt_alpha, length_scales ) :
        self.amplitude     = amplitude
        self.sqrt_alpha    = sqrt_alpha
        self.length_scales = length_scales

    def __call__( self, input_diffs ) :
        scaled_diffs = input_diffs / self.length_scales
        alpha = self.sqrt_alpha ** 2
        return self.amplitude ** 2 * ( 1.0 + 0.5 * scipy.sum( scaled_diffs ** 2, 2 ) / alpha ) ** ( -alpha )

    @property
    def hyperparameters( self ) :
        return collections.deque( [ self.amplitude, self.sqrt_alpha ] + self.length_scales.tolist() )

    def take_hyperparameters( self, deque ) :
        self.amplitude  = deque.popleft()
        self.sqrt_alpha = deque.popleft()
        for i in range( len( self.length_scales ) ) :
            self.length_scales[ i ] = deque.popleft()
