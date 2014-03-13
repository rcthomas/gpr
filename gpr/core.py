
import collections

class GaussianProcess ( object ) :

    def __init__( self, mean_function, covariance_function ) :
        self.mean_function       = mean_function
        self.covariance_function = covariance_function

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
