
from interface import DataSet

class BasicDataSet ( DataSet ) :

    def __init__( self, inputs, responses, covariance ) :
        self._inputs     = inputs
        self._responses  = responses
        self._covariance = covariance

    @property
    def inputs( self ) :
        return self._inputs

    @property
    def responses( self ) :
        return self._responses

    @property
    def covariance( self ) :
        return self._covariance
