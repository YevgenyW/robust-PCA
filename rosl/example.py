import os, sys, warnings
import scipy.io
import numpy as np
import pyrosl
# from hyperspy import signals
# import hyperspy.hspy as hs

"""

     Example usage of pyROSL
     Last modified: 17/11/2015
     
"""

# Parameters to create dataset
n = 2000
m = 2000
rank = 5     # Actual rank
p = 0.1      # Percentage of sparse errors

# Parameters for ROSL
regROSL = 0.03
estROSL = 10

# Parameters for ROSL+
regROSLp = 0.05
estROSLp = 10
samplesp = (250, 250)

#####################################
# No need to modify below this line #
#####################################


# Basis
U = np.random.randn(n, rank)
V = np.random.randn(m, rank)
R = np.dot(U, np.transpose(V))

# Sparse errors
E = -1000 + 1000 * np.random.rand(n, m)
E = np.random.binomial(1, p, (n, m)) * E

# Add the errors
X = R + E

# Run the sub-sampled version
print (' ')
ss_rosl = pyrosl.ROSL( 
    method='subsample',
    sampling = samplesp,
    rank = estROSLp,
    reg = regROSLp,
    iters = 100,
    verbose = True
)
ss_loadings = ss_rosl.fit_transform(X)

# Run the full ROSL algorithm
print (' ')
full_rosl = pyrosl.ROSL(
    method = 'full',
    rank = estROSL,
    reg = regROSL,
    verbose = True
   )
full_loadings = full_rosl.fit_transform(X)

# Output some numbers
ssmodel = np.dot(ss_loadings, ss_rosl.components_)
fullmodel = np.dot(full_loadings, full_rosl.components_)

error1 = np.linalg.norm(R - ssmodel, 'fro') / np.linalg.norm(R, 'fro')
error2 = np.linalg.norm(R - fullmodel, 'fro') / np.linalg.norm(R, 'fro')
error3 = np.linalg.norm(fullmodel - ssmodel, 'fro') / np.linalg.norm(fullmodel, 'fro')
print ('---')
print ('Subsampled ROSL+ error: %.5f' % error1)
print ('Full ROSL error:        %.5f' % error2)
print ('ROSL/ROSL+ comparison:  %.5f' % error3)
print ('---')
