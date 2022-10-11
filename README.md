# Contra-X

This repo contains implementation of the Contra-X models in the paper \
Whodunit? Learning to Contrast for Authorship Attribution, AACL-IJCNLP 2022 [[paper](https://arxiv.org/abs/2209.11887)]

## Get started

First, clone the repo and create a conda environment. Install the packages in `requirements.txt` with `pip`

```
git clone https://github.com/BoAi01/Contra-X.git
cd Contra-X
conda create -n contrax 
conda activate contrax
pip install -r requirements.txt 
```

You may need to manually install `torch` based on your cuda version. Instructions can be found on its [official website](https://pytorch.org/get-started/locally/). Our experimental results are obtained with `torch==1.12.1+cu116`.

Then download the dataset. 
```
python prepare_datasets.py
```
The datasets have been preprocessed for training. In particular, the orignal `TuringBench` dataset can be found [here](https://turingbench.ist.psu.edu/).

## Training

Command line arguments are specified in `main.py`. Here are two example commands that start the training jobs on the blog10 and the TuringBench datasets respectively: 
```
python main.py --dataset blog --id blog10 --gpu 0 --tqdm True --authors 10 \
--epochs 8 --model microsoft/deberta-base
```
```
python main.py --dataset turing --id turingbench --gpu 0 --tqdm True --epochs 10 \
--model microsoft/deberta-base 
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


