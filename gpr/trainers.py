
import sys

import scipy
import scipy.optimize

from interface import Trainer

# Much more work is needed here.  I am not sure I want to do it this way.

class BasinHoppingTrainer ( Trainer ) :

    def __init__( self, niter = 100 ) :
        self.niter = niter

    def __call__( self, model ) :
        def func( hyperparameters, model ) :
            model.hyperparameters = hyperparameters
            log_likelihood = model.log_p
            sys.stderr.write( "%s %s\r" % ( list( model.hyperparameters ), log_likelihood ) )
            return -log_likelihood
        result = scipy.optimize.basinhopping( func, list( model.hyperparameters ), niter = self.niter, minimizer_kwargs = { "args" : ( model, ) } )
        sys.stderr.write( "\n" )
        model.hyperparameters = result.x
        return result
