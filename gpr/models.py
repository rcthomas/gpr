
import collections

import scipy
import scipy.linalg

from interface import Model

class BasicModel ( Model ) :

    def __init__( self, gaussian_process, training_data, update = True ) :
        self.gaussian_process = gaussian_process
        self.training_data    = training_data
  
        training_size       = len( self.training_data )
        self._input_diffs   = self.training_data.inputs[ None, : ] - self.training_data.inputs[ :, None ]
        self._gram          = scipy.zeros( ( training_size, training_size ) )
        self._log_gram_det  = None
        self._inv_gram      = scipy.zeros_like( self._gram )
        self._residuals     = scipy.zeros( training_size )
        self._inv_gram_resp = scipy.zeros( training_size )

        if update :
            self._update()

    @property
    def log_p( self ) :
        return -0.5 * ( scipy.dot( self._residuals, self._inv_gram_resp ) + self._log_gram_det + len( self.training_data ) * scipy.log( 2.0 * scipy.pi ) )

    @property
    def hyperparameters( self ) :
        deque = self.gaussian_process.mean_function.hyperparameters
        deque.extend( self.gaussian_process.covariance_function.hyperparameters )
        return deque

    @hyperparameters.setter
    def hyperparameters( self, iterable ) :
        deque = collections.deque( iterable )
        self.gaussian_process.mean_function.take_hyperparameters( deque )
        self.gaussian_process.covariance_function.take_hyperparameters( deque )
        self._update()

    def __call__( self, inputs, return_covariance = True ) :
        kstar     = self.gaussian_process.covariance_function( self.training_data.inputs[ :, None ] - inputs[ None, : ] )
        responses = self.gaussian_process.mean_function( inputs ) + scipy.dot( kstar.T, self._inv_gram_resp )
        if not return_covariance :
            return responses
        else :
            ksstar     = self.gaussian_process.covariance_function( inputs[ :, None ] - inputs[ None, : ] )
            covariance = ksstar - scipy.dot( kstar.T, scipy.dot( self._inv_gram, kstar ) )
            return responses, covariance

    def _update( self ) :
        self._gram          = self.gaussian_process.covariance_function( self._input_diffs ) + self.training_data.covariance
        cholesky, lower     = scipy.linalg.cho_factor( self._gram, lower = True )
        self._log_gram_det  = 2.0 * scipy.sum( scipy.log( cholesky.diagonal() ) )
        self._inv_gram      = scipy.linalg.cho_solve( ( cholesky, lower ), scipy.eye( len( self.training_data ) ) )
        self._residuals     = self.training_data.responses - self.gaussian_process.mean_function( self.training_data.inputs )
        self._inv_gram_resp = scipy.dot( self._inv_gram, self._residuals )
