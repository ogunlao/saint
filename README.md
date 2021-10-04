# SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pretraining

![saint architecture](saint.png)

Paper Reference: https://arxiv.org/abs/2106.01342

> NB: This implementation uses [Pytorch-lightning](https://pytorch-lightning.readthedocs.io/en/latest/) and [Hydra](https://hydra.cc/) for configuration. For an earlier release of this repo, check the branch, [saint-orig](https://github.com/ogunlao/saint/tree/saint-orig)

We got AUROC of 92.9% on bank dataset with initial experiments. More can be done in terms of hyperparameter sweep.

The code currently work for binary and multiclass classification tasks. Regression not supported at the moment.

Major modules implemented in the code

- Saint Transformer
- Saint Intersample Transformer
- Embeddings for tabular data
- Mixup
- CutMix
- Contrastive Loss
- Denoising Loss

## How to use code

### Process your dataset in the following format:

- Add cls column to dataset. 'cls' column has to be the first column as mentioned in paper
- Apply z-transform to numerical columns
- Label encode categorical columns
- Concatenate cat and num columns, with cat columns coming first, then numerical ones
- Calculate the number of categorical columns (including 'cls' column), and numerical columns. Add to config file as 'no_cat' and 'no_num'
- Calculate the number of categories in each categorical columns, as a list. Add to config file as 'cats'. 'cls' column has 1 category
- Sample function `preprocess_bank` can be used to preprocess bank dataset. It can be found in `src > dataset.py`
- Save files in train, val and test csv in `data` folder

### Clone the repository

```git
git clone https://github.com/ogunlao/saint.git
```

### Setup a new environment using `requirements.txt` in repo

```python
pip3 install -r requirements.txt 
```

### Setup configuration in `configs` directory

The base config can be found at `configs > config.yaml`. Other configs related to the data and experiment can also be found in the `configs` dir

### Run `python main.py` with command-line arguments or with edited config file

Examples

1. To train saint_i model in self-supervised mode using bank dataset, run;

```bash
python main.py experiment=self-supervised \
experiment.model=saint_i data=bank_ssl
```

2. To train saint model in supervised mode using bank dataset, run;

```bash
python main.py \
experiment=supervised \
experiment.model=saint \
data=bank_sup
```

3. To make prediction using saint_s model in supervised mode using bank dataset, run;

```bash
python predict.py experiment=predict \
experiment.model=saint_s data=bank_sup \
experiment.pretrained_checkpoint=["PATH_TO_CKPT"]
```

Data Prerocessing can be done in a similar manner for other classification dataset

### Contributors

- [Ahmed A. Elhag](https://github.com/Ahmed-A-A-Elhag)
- [Aisha Alaagib](https://github.com/AishaAlaagib)
- [Amina Rufai](https://github.com/Aminah92)
- [Amna Ahmed Elmustapha](https://github.com/AMNAALMGLY)
- [Jamal Hussein](https://github.com/engmubarak48)
- [Mohammedelfatih Salah](https://github.com/mohammedElfatihSalah)
- [Ruba Mutasim](https://github.com/ruba128)
- [Sewade Olaolu Ogun](https://github.com/ogunlao)

(names in alphabetical order)
