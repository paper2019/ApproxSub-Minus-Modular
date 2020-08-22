# ApproxSub-Minus-Modular
Code and data for the paper "Multi-objective Evolutionary Algorithms are Still Good: Maximizing Monotone Approximately Submodular Minus Modular Functions"

This package consists of 2 experiments: Bayesian Experimental Design and Directed Vertex Cover.

1. The folder 'Bayesian_Experimental_Design' corresponds to the experiment of Bayesian Experimental Design. We implement this experiment on two datasets, boston_housing and segment, respectively.

For the boston_housing dataset in the sub-folder 'boston_housing', in order to generate different lower bounds of the submodularity ratio $\gamma$, we set $\sigma$ variously.

The sub-folder '3d' corresponds to the case $\sigma$ =3d, containing the python file and the corresponding data. In this case, $\gamma$=0.456.
The sub-folder '4d' corresponds to the case $\sigma$ =4d, containing the python file and the corresponding data. In this case, $\gamma$=0.567.
The sub-folder '7d' corresponds to the case $\sigma$ =7d, containing the python file and the corresponding data. In this case, $\gamma$=0.800.

For the segment dataset in the sub-folder 'segment', in order to generate different lower bounds of the submodularity ratio $\gamma$, we set $\sigma$ variously.

The sub-folder '6d' corresponds to the case $\sigma$ =6d, containing the python file and the corresponding data. In this case, $\gamma$=0.281.
The sub-folder '11d' corresponds to the case $\sigma$ =11d, containing the python file and the corresponding data. In this case, $\gamma$=0.621.
The sub-folder '20d' corresponds to the case $\sigma$ =20d, containing the python file and the corresponding data. In this case, $\gamma$=0.840.

These sub-folders actually share the same python file 'Bayesian_Experimental_Design.py', containing five algorithms: Distorted Greedy (DG), Stochastic Distorted Greedy (SDG), GSEMO, GSEMO_{g-c} and Multiple SDG (Multi-SDG). The subtle differences are that each python file uses its own $\gamma$ setting and data set. Users can run the python file directly without changing the relative data path. Note that the three files, '__init__.py', 'nsga2.py' and 'test_nsga2.py' are the implementation of NSGA-II, and the user can run 'test_nsga2.py' directly. 

2. The folder 'Directed_Vertex_Cover' corresponds to the experiment of Directed Vertex Cover.

The sub-folder 'email-Eu-core' corresponds to the data set 'email-Eu-core.txt'.
The sub-folder 'frb45-21-mis' corresponds to the data set 'frb45-21-mis'.
The sub-folder ''frb53-24-mis' corresponds to the data set 'frb53-24-mis'.

Each sub-folder contains the python file 'Directed_Vertex_Cover.py' and its corresponding data set. In fact, they share the same five algorithms: DG, SDG, GSEMO, GSEMO_{g-c} and Multi-SDG. But each python file needs to deal with different data sets. Users can run the python file directly without changing the relative data path. Note that the three files, '__init__.py', 'nsga2.py' and 'test_nsga2.py' are the implementation of NSGA-II, and the user can run 'test_nsga2.py' directly. 
