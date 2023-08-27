# Density Ratio Estimation for Selection Experiments

This repo includes the code for the analyses presented in the paper

A. Busia and J. Listgarten. Model-based differential sequencing analysis. *bioRxiv*, 2023.

which quantify high-throughput selection experiments using model-based enrichment (MBE)---a density ratio estimation (DRE) approach for estimating and/or predicting log-enrichment from sequencing data.


Key components include: MBE is implemented using linear, fully-connected neural network, and convolutional neural network model architectures defined in ```modeling.py```, which are trained and evaluted using ```run_models.py``` and ```evaluate_models.py```. See the [MBE package](https://github.com/apbusia/model_based_enrichment) for a more general implementation. Additional analyses of negative selection simulations are implemented in ```negative_selection.py``` and the scripts in ```plotting``` generate the visuals presented in the paper's main text and supplementary information. Simulated libraries and sequencing datasets were generated using the ```simulate_(...).py``` scripts, ```simlord_from_counts.py```, and ```add_random_noise.py```.
