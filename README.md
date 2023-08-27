# Density Ratio Estimation for Selection Experiments

This repo includes the code for the analyses presented in the paper

A. Busia and J. Listgarten. Model-based differential sequencing analysis. *bioRxiv*, 2023.

which use model-based enrichment (MBE)---a density ratio estimation (DRE) approach for estimating and/or predicting log-enrichment from sequencing data---to quantify high-throughput selection experiments. See the [MBE package](https://github.com/apbusia/model_based_enrichment) for a more general implementation of MBE.


MBE is implemented using the linear, fully-connected neural network, and convolutional neural network model architectures defined in ```modeling.py```, which are trained and evaluted using ```run_models.py``` and ```evaluate_models.py```. In addition to the model evaluations in ```evaluate_models.py```, ```negative_selection.py``` implements analyses of negative selection simulations and the scripts in ```plotting``` generate the plots presented in the main text and SI of the paper. Simulated libraries and sequencing datasets were generated using the ```simulate_(...).py``` scripts, ```simlord_from_counts.py```, and ```add_random_noise.py```.
