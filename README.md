# SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pretraining

Unofficial Implementation By Sewade Ogun, Aisha Alagib, Jamal Hussein, ...

AUROC of 92.9% on bank dataset

Major modules implemented in the code

- Saint Transformer
- Saint Intersample Transformer
- Embeddings for tabular data
- Mixup
- CutMix
- Contrastive Loss
- Denoising Loss

## How to use code

1. Process dataset in the following format:
    - Add cls column to dataset. 'cls' column has to be the first column as mentioned in paper
    - Apply z-transform to numerical columns
    - Label encode categorical columns
    - Concatenate cat and num columns, with cat columns coming first, then numerical ones
    - Calculate the number of categorical columns \(including 'cls' column\), and numerical columns. Add to config file as 'no_cat' and 'no_num'
    - Calculate the number of categories in each categorical columns, as a list. Add to config file as 'cats'. 'cls' column has 1 category
    - Sample of script to preprocess file is in `dataset.py` file  in `src` folder

1. Run `python main.py` with command-line arguments or with properly edited config file.


### TODO:

1. Evaluate on more datasets
1. Optimize the embedding layer for fast retrieval of embeddings
1. Improve documentation
