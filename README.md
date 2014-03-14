gpr
===

Simple Python package for Gaussian process (GP) regression problems.  Depends
on scipy.  Just do

    python setup.py install

To have python run setup.py which installs.

Basic Workflow
--------------

The basic workflow this package supports looks like this.  More details
are further down in this README.

* Get data.
* Wash hands.
* Put data into instantiation of a subclass of gpr.interface.DataSet.
* Construct a mean function and a covariance function.
* Construct a GP taking the mean and covariance functions.
* Construct a model based on the GP and the (training) data.
* Construct a trainer and train the model.
* Predict data.

Standardization of the training data and test data are up to you, but
the DataSet class may be able to handle this for you.  

Imports
-------

This package exposes useful interfaces but also provides a number of 
pre-built parts to combine out of the box.

    import gpr

provides the main Gaussian process class.

    import gpr.data_sets

provides pre-built data set classes.  Users should probably subclass
gpr.interface.DataSet instead of using a pre-built data set class for 
production.  This could also handle standardization.  Next there is

    import gpr.functions

which contains mean functions, covariance functions, and operator 
functions that can be used to combine them together.  Then,

    import gpr.models

provides pre-built model classes.  Users may wish to construct their
own by subclassing gpr.interface.Model.  Finally,

    import gpr.trainers

provides trainers for models.  This maximizes the log likelihood of 
the model.  How this is done may change in the future.

Data Set
--------

Now let's create a fake data set using one of the pre-built data set
classes.  Just a basic data set.  We will construct it from a sine
function plus a linear function with a dash of noise.

    inputs     = scipy.linspace( -1.0, 1.0 )
    responses  = 1.0 + inputs + scipy.sin( 2.5 * scipy.pi * inputs ) + 
        scipy.random.normal( scale = 0.1, size = len( inputs ) )
    covariance = 0.01 * scipy.eye( len( inputs ) )
    
    data_set = gpr.data_sets.BasicDataSet( inputs[ :, None ], responses, covariance )

Note that inputs here is a 2D array, each row corresponds to one 
independent variable.  This is a 1D input example, but you can do
multi-dimensional input problems with this package.

Functions
---------

There are mean functions, covariance functions, and operator functions
for combining them.  Don't combine mean functions with covariance 
functions using operator functions.  Mean functions can be combined
with mean functions, and covariance functions combined with other
covariance functions, but don't mix them.

Try a polynomial mean function and squared exponential covariance 
plus noise.

    mean_response = data_set.responses.mean()
    std_response  = data_set.responses.std()
    input_spacing = scipy.mean( inputs[ 1 : ] - inputs[ : -1 ] )

    mean_function = gpr.functions.PolynomialMean( [ 1.0, mean_response ] )
    covariance_1  = gpr.functions.IsotropicSquaredExponentialCovariance( std_response, input_spacing )
    covariance_2  = gpr.functions.GaussianWhiteNoiseCovariance( 0.01 * std_response )
    cov_function  = gpr.functions.Sum( covariance_1, covariance_2 )

Note the mean and covariance functions are initialized with some 
hyperparameter settings.

Gaussian Process
----------------

By itself a Gaussian process is kind of useless but it is possible to
generate functions from it (i.e., treat it like a prior over functions).

    gaussian_process = gpr.GaussianProcess( mean_function, cov_function )

A Gaussian process is just our mean function and the covariance function.

Model
-----

The model is a Gaussian process plus the training data set.

    model = gpr.models.BasicModel( gaussian_process, data_set )

All too easy.

Trainer
-------

This gets the model in shape for the big game.  It's kind of stupidly
defined and I need a better interface.

    trainer = gpr.trainers.BasinHoppingTrainer()
    print trainer( model )

Predictions
-----------

Now we are ready to predict.  We can see what different parts of our 
trained model actually look like.  For instance, let's separate the 
Gaussian white noise piece out and just look at the squared exponential.

    pred_gaussian_process = gpr.GaussianProcess( mean_function, covariance_1 )
    pred_model            = gpr.models.BasicModel( pred_gaussian_process, data_set )
    
    pred_responses, pred_covariance = pred_model( data_set.inputs )
    
    pred_sigma = scipy.sqrt( pred_covariance.diagonal() )

    # Plot.

    for i in range( 1, 4 ) :
        lower = pred_responses - i * pred_sigma
        upper = pred_responses + i * pred_sigma
        plt.fill_between( inputs, lower, upper, alpha = 0.1 )
    plt.plot( inputs, pred_responses )
    plt.scatter( inputs, responses )
    plt.show()

The Code
--------

Refer to the example codes.  The code presented here is based on the 1D example
script.
