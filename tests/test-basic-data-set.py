
import sys
import unittest

import scipy

sys.path.insert( 0, ".." )
sys.path.insert( 0, "../build/lib" )
import gpr.data_sets

class TestBasicDataSet ( unittest.TestCase ) :

    def setUp( self ) :
        self.inputs     = scipy.random.uniform( size = ( 10, 2 ) )
        self.responses  = scipy.random.uniform( size = 10 )
        self.covariance = scipy.diag( scipy.random.uniform( size = 10 ) ** 2 )

    def test_init( self ) :
        data_set = gpr.data_sets.BasicDataSet( self.inputs, self.responses, self.covariance )
        self.assertTrue( ( data_set.inputs     == self.inputs     ).all() )
        self.assertTrue( ( data_set.responses  == self.responses  ).all() )
        self.assertTrue( ( data_set.covariance == self.covariance ).all() )

if __name__ == "__main__" :
    unittest.main()
