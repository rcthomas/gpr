
# Operator functions are used to combine other functions.

from interface import Function

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
