import os
import tarfile

import gdown

datasets = {
    'contrax_datasets.tar': 'https://drive.google.com/uc?id=1T3VgMe-dCy5QVI7b1K2KdfL-2e2gq2Rn'
}

dataset_path = 'datasets'
os.makedirs(dataset_path, exist_ok=True)

if __name__ == "__main__":
    for name, link in datasets.items():
        if name in os.listdir(dataset_path):
            continue
        gdown.download(link, name, quiet=False)

    tar = tarfile.open(list(datasets.keys())[0])
    tar.extractall(path='datasets')
    tar.close()
