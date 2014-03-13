
import collections

class GaussianProcess ( object ) :

    def __init__( self, mean_function, covariance_function ) :
        self._mean_function       = mean_function
        self._covariance_function = covariance_function

    @property
    def mean_function( self ) :
        return self._mean_function

    @property
    def covariance_function( self ) :
        return self._covariance_function

    @property
    def hyperparameters( self ) :
        deque = self.mean_function.hyperparameters
        deque.extend( self.covariance_function.hyperparameters )
        return deque

    @hyperparameters.setter
    def hyperparameters( self, iterable ) :
        deque = collections.deque( iterable )
        self.mean_function.take_hyperparameters( deque )
        self.covariance_function.take_hyperparameters( deque )
