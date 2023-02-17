This is a re-implementation of the original [DeepGuide](https://github.com/dDipankar/DeepGuide) showcased in Baisya, D., Ramesh, A., Schwartz, C. <em>et al.</em> Genome-wide functional screens enable the prediction of high activity CRISPR-Cas9 and -Cas12a guides in <em>Yarrowia lipolytica</em>. Nat Commun 13, 922 (2022). https://doi.org/10.1038/s41467-022-28540-0

This iteration of DeepGuide allows a higher degree of versatility via a configuration file. It also allows pre-training and training the model from scratch by switching modes in config.yaml. Experiments done in our [acCrispr](https://www.biorxiv.org/content/10.1101/2022.07.12.499789v1.full.pdf) were done using this iteration.


## Pre-requisites
0. Training DeepGuide is orders of magnitude faster on a machine with an Nvidia GPU. 
1. DeepGuide is written in Python. To run it, get Anaconda for your specific operating system [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html)
2. Clone this repository by either clicking on the green Code on the top right and clicking "Download ZIP," or downloading [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and then `$ git clone https://github.com/AmirUCR/deepguide_reborn.git` in your desired directory.
3. Create a conda environment and let conda install the required packages via the command below. Answer yes to the installation prompt.
```
$ conda create -n deepguide --file requirements.yml
```
4. Switch to the newly-created environment like so: `conda activate deepguide`
5. Run the test input like so: `python src/main.py` and check the output in `data/output/example_run/example_test_cas9_seq_dg1_28nt_predicted_scores.csv`
6. Refer to the config.yaml file where each customizable option of DeepGuide Reborn is explained in detail.

## Citation

**Genome-wide functional screens enable the prediction of high activity CRISPR-Cas9 and -Cas12a guides in *Yarrowia lipolytica***. Dipankar Baisya, Adithya Ramesh, Cory Schwartz, Stefano Lonardi, and Ian Wheeldon. Nature Communication, 2022

[![DOI](https://zenodo.org/badge/404852665.svg)](https://zenodo.org/badge/latestdoi/404852665)
