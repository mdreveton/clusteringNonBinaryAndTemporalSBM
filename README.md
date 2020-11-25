# MarkovSBM

This is the version of the code that was used to produce the numerical results of our paper
"Estimation of Static Community Memberships from Temporal Network Data" (arXiv:2008.04790)
by Konstantin Avrachenkov, Maximilien Dreveton, Lasse Leskelä


Code by Maximilien Dreveton.



# Content

File MarkovSBM.py: functions to generate the adjacency matrix of a Markov SBM
File online_likelihood_clustering_known_parameters.py : contain the likelihood algorithm (Algorithm 2 of the paper)
File online_likelihood_clustering_unknown_parameters.py : contain the likelihood algorithm when interaction parameters are unknown and need to be estimated (Algorithm 3 of the paper)
File baseline_clustering_algorithms.py : contain Algorithms 5, 6, as well as SpectralClustering and the algorithm of the paper Jing  Lei. Tail  bounds  for  matrix  quadratic  forms  and bias  adjusted  spectral  clustering  in  multi-layer  stochastic  block  models. 

The other files (named plot_figure_...) are self-contained and can be used to plot the different Figures in the paper.
Note that the parameters in those files are set to the one used in the paper's Figures, but can be modified by the user.
The estimated computation time is given for information.
