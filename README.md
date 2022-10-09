# CS4248-Authorship-Attribution

## Run the code

To understand all relavant command line arguments, see `main.py`. One example run: </p>
```python main.py --dataset ccat50 --id 0 --tqdm True```

## Env setup

PyTorch needs to be compiled with the correct CUDA version. Example for CUDA 11: </p>
```conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch ``` </p>
For more info, see  [here](https://pytorch.org/).  </p>

Manually install APEX:

```
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

For the other depdencies, manual installation is required.

## Training

One-time preparation of dataset </p>
``` python prepare_dataset.py ```

Start training with</p>
``` python main.py --dataset <dataset name in ['imdb62', 'enron', 'imdb', 'blog']>  --gpu <gpu indices, optional> --samples-per-author <# of samples per author> ```

## Pre-trained model training

- Change the `model_name` argument in the function call of `train_bert` in `train.py` to that of the desired
  model (`bert-base-cased`, `roberta-base`, `microsoft/deberta-base`, `gpt2`, or `xlnet-base-cased`).

## Ensemble training

- Download the pre-trained checkpoints for the various models
  from [Google Drive](https://drive.google.com/drive/folders/1g0_-YhqvgCo6Z6x4tBu4Cwt-jp5orbcw).
- Update the paths to the respective models in the function call of `train_ensemble` in `train.py`.
