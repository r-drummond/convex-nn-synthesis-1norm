# convex-nn-synthesis-1norm
Code to reproduce the results of "Convex neural network synthesis for robustness in the 1-norm" by Drummond, Guiver and Turner. Presented at the Learning for Decision and Control Conference 2024. 

The code solves a semi-definite programme to generate neural network weights and biases that are close to an original network but are more robust. The main application considered here is to generate robust approximation of model predictive control policies. Tuning the tol_eps parameter allows to place more emphasis on either robustess (tol_eps = large) or accuracy (tol_eps = small).

The code uses YALMIP to pose the semi-definite programmes and MOSEK to solve them (however, other compatible SDP solvers could also be used). 
