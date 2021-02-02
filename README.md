# NICE-features
Algorithms used to compute N-body features and determine irreducible features.

Algorithm 1: 
This algorithm computes N-Body features using mainly the Python library Sympy. 
It is a basic translation of the NICE-recursion-formula. 
Coefficient values are in symbolic form and thus exact.
Features are ordered with respect to l's and n's, in order to avoid repetitions.

Algorithm 2:
This algorithm computes N-Body features using mainly the Python library Sympy. 
In addition to the NICE-recursion-formula, properties of inversion symmetry are used to reduce computational cost. 
Coefficient values are in symbolic form and thus exact.
Features are ordered with respect to l's and n's, in order to avoid repetitions.

Algorithm 3:
This algorithm computes N-Body features using mainly the Python libraries Numpy and Scipy. 
Computation of features relies on linear algebra calculations and features appears in the form of vectors.
Coefficient values are numerical approximations, tolerance must be given.
Features are ordered with respect to l's and n's, in order to avoid repetitions.

Algorithm 4:
This algorithm computes a list of irreducible features, i.e. irreducible under linear and polynomial decomposition.
In a first step, features must be computed using algorithm 1 before running this algorithm.
