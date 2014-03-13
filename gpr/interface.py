
class Function ( object ) :

    def __call__( self, inputs ) :
        raise NotImplementedError

    @property
    def hyperparameters( self ) :
        raise NotImplementedError

    def take_hyperparameters( self, deque ) :
        raise NotImplementedError

class DataSet ( object ) :

    def __len__( self ) :
        return len( self.inputs )

    def __nonzero__( self ) :
        return len( self ) > 0

    @property
    def inputs( self ) :
        raise NotImplementedError

    @property
    def responses( self ) :
        raise NotImplementedError

    @property
    def covariance( self ) :
        raise NotImplementedError

class Model ( object ) :

    def __call__( self, inputs, return_covariance = False ) :
        raise NotImplementedError

    @property
    def log_p( self ) :
        raise NotImplementedError

    @property
    def hyperparameters( self ) :
        raise NotImplementedError

    @hyperparameters.setter
    def hyperparameters( self, iterable ) :
        raise NotImplementedError

class Trainer ( object ) :

    def __call__( self, model ) :
        raise NotImplementedError
