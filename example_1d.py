#!/usr/bin/env python

# Example to show how you can use GPR.  Note that data should usually
# be standardized in some way.  That is up to you.  Also this is a 
# multidimensional optimization problem, so one needs to watch out
# for the usual things like local maxima in the likelihood function
# when training.  The usual caveats.

import matplotlib.pyplot as plt
import scipy

import gpr
import gpr.data_sets
import gpr.functions
import gpr.models
import gpr.trainers

# Create a fake data set.  In practice the user should subclass
# gpr.interface.DataSet and this could handle standardization.

inputs     = scipy.linspace( -1.0, 1.0 )
responses  = 1.0 + inputs + scipy.sin( 2.5 * scipy.pi * inputs ) + scipy.random.normal( scale = 0.1, size = len( inputs ) )
covariance = 0.01 * scipy.eye( len( inputs ) )

data_set = gpr.data_sets.BasicDataSet( inputs[ :, None ], responses, covariance )

# Try a polynomial mean function and squared exponential covariance 
# plus noise.

mean_function = gpr.functions.PolynomialMean( [ 1.0, data_set.responses.mean() ] )
covariance_1  = gpr.functions.IsotropicSquaredExponentialCovariance( data_set.responses.std(), scipy.mean( inputs[ 1 : ] - inputs[ : -1 ] ) )
covariance_2  = gpr.functions.GaussianWhiteNoiseCovariance( 0.01 * data_set.responses.std() )
cov_function  = gpr.functions.Sum( covariance_1, covariance_2 )

# Gaussian process is a mean and covariance function.

gaussian_process = gpr.GaussianProcess( mean_function, cov_function )

# Model is Gaussian process and training data.

model = gpr.models.BasicModel( gaussian_process, data_set )

# A trainer trains the model.

trainer = gpr.trainers.BasinHoppingTrainer()

print trainer( model )

# Prediction model for the underlying signal based on trained components.

pred_gaussian_process = gpr.GaussianProcess( mean_function, covariance_1 )
pred_model            = gpr.models.BasicModel( pred_gaussian_process, data_set )

pred_responses, pred_covariance = pred_model( data_set.inputs )

pred_sigma = scipy.sqrt( pred_covariance.diagonal() )

# Plot.

for i in range( 1, 4 ) :
    plt.fill_between( inputs, pred_responses - i * pred_sigma, pred_responses + i * pred_sigma, alpha = 0.1 )
plt.plot( inputs, pred_responses )
plt.scatter( inputs, responses )
plt.show()
