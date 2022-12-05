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
@inproceedings{ai-etal-2022-whodunit,
    title = "Whodunit? Learning to Contrast for Authorship Attribution",
    author = "Ai, Bo  and
      Wang, Yuchen  and
      Tan, Yugin  and
      Tan, Samson",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-main.84",
    pages = "1142--1157",
    abstract = "Authorship attribution is the task of identifying the author of a given text. The key is finding representations that can differentiate between authors. Existing approaches typically use manually designed features that capture a dataset{'}s content and style, but these approaches are dataset-dependent and yield inconsistent performance across corpora. In this work, we propose to learn author-specific representations by fine-tuning pre-trained generic language representations with a contrastive objective (Contra-X). We show that Contra-X learns representations that form highly separable clusters for different authors. It advances the state-of-the-art on multiple human and machine authorship attribution benchmarks, enabling improvements of up to 6.8{\%} over cross-entropy fine-tuning. However, we find that Contra-X improves overall accuracy at the cost of sacrificing performance for some authors. Resolving this tension will be an important direction for future work. To the best of our knowledge, we are the first to integrate contrastive learning with pre-trained language model fine-tuning for authorship attribution.",
}
```


