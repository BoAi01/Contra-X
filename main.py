import argparse
import os
import random
import warnings

import numpy
import torch

from training import train_bert
from utils import load_dataset_dataframe, build_train_test

random.seed(0)
numpy.random.seed(0)
torch.manual_seed(0)

if __name__ == '__main__':
    datasets = ['imdb62', 'blog', 'turing']
    parser = argparse.ArgumentParser(description=f'Training models for datasets {datasets}')
    parser.add_argument('--dataset', type=str, help='dataset used for training', choices=datasets)
    parser.add_argument('--id', type=str, default='0', help='experiment id')
    parser.add_argument('--gpu', type=str, help='the cuda devices used for training', default="0,1,2,3")
    parser.add_argument('--tqdm', type=bool, help='whether tqdm is on', default=False)
    parser.add_argument('--authors', type=int, help='number of authors', default=None)
    parser.add_argument('--samples-per-auth', type=int, help='number of samples per author', default=None)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model', type=str, default='microsoft/deberta-base')
    parser.add_argument('--train-ensemble', type=bool, default=False)
    parser.add_argument('--coe', type=float, default=1)

    # dataset - num of authors mapping
    default_num_authors = {
        'imdb62': 62,
        'blog': 50,
        'turing': 20
    }

    # parse args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    source = args.dataset
    num_authors = args.authors if args.authors is not None else default_num_authors[args.dataset]
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))  # print all args

    # masked classes
    mask_classes = {
        'blog': [],
        'imdb62': [],
        'enron': []
    }

    # load data and remove emails containing the sender's name
    df = load_dataset_dataframe(source)

    if args.authors is not default_num_authors[args.dataset]:
        warnings.warn(f"Number of authors for dataset {args.dataset} is {default_num_authors[args.dataset]}, "
                      f"but got {args.authors} instead. ")

    if args.samples_per_auth is not None:
        warnings.warn(f"Number of samples per author specified as {args.samples_per_auth}, which is a "
                      f"dangerous argument. ")

    limit = num_authors
    print("Number of authors: ", limit)

    # select top N senders and build train and test
    nlp_train, nlp_val, nlp_test = build_train_test(df, source, limit, per_author=args.samples_per_auth, seed=0)

    # train
    if 'enron' in source or 'imdb62' in source or 'blog' in source:
        train_bert(nlp_train, nlp_test, args.tqdm, args.model, 768, args.id, args.epochs, base_bs=8, base_lr=1e-5,
                   mask_classes=mask_classes[args.dataset], coefficient=args.coe, num_authors=num_authors,
                   val_dic=nlp_val)
    elif 'turing' in source:
        train_bert(nlp_train, nlp_test, args.tqdm, args.model, 768, args.id, args.epochs, base_bs=7, base_lr=5e-6,
                   mask_classes=mask_classes[args.dataset], coefficient=args.coe, num_authors=num_authors,
                   val_dic=nlp_val)
    else:
        train_bert(nlp_train, nlp_test, args.tqdm, args.model, 768, args.id, args.epochs, base_bs=8, base_lr=1e-5,
                   mask_classes=mask_classes[args.dataset], coefficient=args.coe, num_authors=num_authors)
