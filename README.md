# DeepGuide Reborn
This is a re-implementation of the original [DeepGuide](https://github.com/dDipankar/DeepGuide) showcased in Baisya, D., Ramesh, A., Schwartz, C. <em>et al.</em> Genome-wide functional screens enable the prediction of high activity CRISPR-Cas9 and -Cas12a guides in <em>Yarrowia lipolytica</em>. Nat Commun 13, 922 (2022). https://doi.org/10.1038/s41467-022-28540-0

This iteration of DeepGuide allows a higher degree of versatility via a configuration file. It also allows pre-training and training the model from scratch by switching modes in config.yaml. Experiments done in our [acCRISPR](https://www.biorxiv.org/content/10.1101/2022.07.12.499789v1.full.pdf) paper were performed using this iteration.

DeepGuide Reborn uses the [Decorator](https://refactoring.guru/design-patterns/decorator) and [Factory](https://refactoring.guru/design-patterns/factory-method) design patterns to manage data preprocessing, machine learning model creation, and output post-processing (such as calculating correlation co-efficients and drawing graphs). Extending DeepGuide's functionality is streamlined.

## Pre-requisites
0. Training DeepGuide is orders of magnitude faster on a machine with an Nvidia GPU. 
1. DeepGuide is written in Python. To run it, get Anaconda for your specific operating system [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html)
2. Clone this repository by either clicking on the green Code on the top right and clicking "Download ZIP," or downloading [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and then `$ git clone https://github.com/AmirUCR/deepguide_reborn.git` in your desired directory.
3. Create a conda environment and activate it.

```
conda create -n deepguide python=3.10 -y

conda activate deepguide
```

4. Install CUDA and cuDNN with conda and pip.
```
conda install -c conda-forge cudatoolkit=11.8.0 -y

pip install nvidia-cudnn-cu11==8.6.0.163
```

5. Configure the system paths.
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

6. Upgrade pip and install TF.

```
pip install --upgrade pip

pip install tensorflow==2.12.*
```

7. Exit the current environment and activate it again. This makes the system paths in step 3 to be initialized.
```
conda deactivate

conda activate deepguide
```

8. Confirm that TF can see the GPU.
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]` on the last line of the output.

9. Install the rest of the dependencies for DeepGuide.

```
conda install pyyaml pandas numpy scikit-learn matplotlib pydot pydotplus biopython -c conda-forge -y
```

10. Run the test input like so: `python src/main.py` and check the output in `data/output/example_run/example_test_cas9_seq_dg1_28nt_predicted_scores.csv`
11. Refer to the config.yaml file where each customizable option of DeepGuide Reborn is explained in detail.

## Citation

**Genome-wide functional screens enable the prediction of high activity CRISPR-Cas9 and -Cas12a guides in *Yarrowia lipolytica***. Dipankar Baisya, Adithya Ramesh, Cory Schwartz, Stefano Lonardi, and Ian Wheeldon. Nature Communication, 2022

[![DOI](https://zenodo.org/badge/404852665.svg)](https://zenodo.org/badge/latestdoi/404852665)
