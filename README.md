# Contra-X

This repo contains implementation of the Contra-X models in the paper \
Whodunit? Learning to Contrast for Authorship Attribution, AACL-IJCNLP 2022 [[arxiv](https://arxiv.org/abs/2209.11887)]

## Get started

Firstly, clone the repo on compute machine with

```git clone https://github.com/BoAi01/Contra-X.git```

Next, create a conda environment with the packages listed in `requirements.txt` installed

```conda create -n contrax --file requirements.txt```

Activate the environment and download the dataset 
```
conda activate contrax
cd Contra-X
python prepare_datasets.py
```
The datasets have been preprocessed. In particular, the orignal `TuringBench` dataset can be found [here](https://turingbench.ist.psu.edu/).

## Training

Command line arguments are specified in `main.py`. One example command to run an experiment on the blog10 dataset: 
```
python main.py --dataset blog --id blog10 --gpu 0 --tqdm True --authors 10 \
--epochs 8 --model microsoft/deberta-base
```

Experiments on other datasets can be run in a similar way. 

## Citation
If you use our implementation in your work, welcome to cite our paper
```bibtex
@article{ai2022whodunit,
  title={Whodunit? Learning to Contrast for Authorship Attribution},
  author={Ai, Bo and Wang, Yuchen and Tan, Yugin and Tan, Samson},
  journal={arXiv preprint arXiv:2209.11887},
  year={2022}
}

```


