from pathlib import Path

import numpy as np
import pandas as pd

from sklearn import preprocessing
from torch.utils.data import Dataset

class DatasetTabular(Dataset):
    """Creates a tabular data set class"""
    def __init__(self, data, y):
        """
        Parameters
        ------------------------
        data: np.array
            contains the features. It's assumed that the features
            are on the order of [cls, categorical features, numerical_features]
        y: np.array
            represents the target variable
        """
        self.data = data.copy()  # bs x n
        self.y = y.copy()        # bs x 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.y[idx]

        return sample, label

    def make_weights_for_imbalanced_classes(self):
        """adopted from https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3"""

        nclasses = len(np.unique(self.y))
        count = np.zeros(nclasses)
        for idx in range(len(self.y)):
            target = self.y[idx]
            count[target] += 1

        N = float(sum(count))
        weight_per_class = N / count
        weight = np.zeros(len(self))
        for idx in range(len(self.y)):
            target = self.y[idx]
            weight[idx] = weight_per_class[target]
        return weight
    
    
def generate_splits(dataset_size, num_supervised_train_data, 
                      validation_split, test_split=0.0, 
                      random_seed=1234, shuffle_dataset=True,):
    """Split dataset indices into train, val, and test 
    for supervised and self-supervised training
    """

    # Create data indices for training and validation splits:
    indices = list(range(dataset_size))

    split_val = int(validation_split * dataset_size)
    split_test = int(test_split * dataset_size)
    
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # should not change for all operations
    test_indices = []
    if test_split:
        test_indices = indices[:split_test] 
    val_indices = indices[split_test : split_test+split_val]
    
    if num_supervised_train_data == 'all':
        sup_train_indices = indices[split_test+split_val: ]
        ssl_train_indices = []
    else:
        num_supervised_train_data = int(num_supervised_train_data)
        sup_train_indices = indices[split_test+split_val: split_test+split_val+num_supervised_train_data]   
        ssl_train_indices = indices[split_test+split_val+num_supervised_train_data: ]

    return sup_train_indices, val_indices, test_indices, ssl_train_indices

 
def preprocess(data, target, cls_token_idx=0):
    """preprocess csv data set and compute the statistics of the data

    Args:
        data (pd.Dataframe): Dataframe of tabular data
        target (str, pd.Dataframe): target column name in the data or the dataframe of the target.
        cls_token_idx (int): index to insert the "cls" token in the dataframe. Defaults to 0

    Returns:
        tuple: tuple containing:
        - data, 
        - labels
        - total_num_of_categorical_columns
        - total_num_of_numerical_columns
        - array of number of categories in each categorical column, \
            in order or appearance in the dataframe. 
    """    
    
    data = data.copy()
    if isinstance(target, str):
        target = data[[target]]
        data = data.drop(columns=[target])

    # add the cls token
    data.insert(loc=cls_token_idx, column='cls', value='cls')

    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    num_cols = [col for col in data.columns if col not in cat_cols]

    # z-transform
    num_data = data[num_cols].copy()
    num_data = (num_data-num_data.mean())/num_data.std()

    cat_data = data[cat_cols].copy()
    
    # fill missing
    num_data = num_data.fillna(-99999)
    cat_data[pd.isnull(cat_data)]  = 'NaN'

    # label encode
    labelencode = preprocessing.LabelEncoder()
    cat_data = cat_data.apply(labelencode.fit_transform)

    # Note that categorical columns come first
    new_data = pd.concat([cat_data.astype(np.int32), num_data.astype(np.float32)], axis=1)

    if target.values.dtype not in ['int32', 'int64', 'float32', 'float64', 'int']:
        labels = labelencode.fit_transform(target)
        labels = pd.DataFrame(labels, columns=target.columns)
    else:
        labels = target
    
    # calculate number of categories in each categorical column
    cats = []
    for cat in cat_data.columns:
        cats.append(len(pd.unique(new_data[cat])))

    return new_data, labels, len(num_data.columns), len(cat_data.columns), cats
    
    
def generate_dataset(train_csv_path, val_csv_path, 
                     test_csv_path=None, train_y_csv_path=None, 
                     val_y_csv_path=None, test_y_csv_path=None,):
    
    train_df = pd.read_csv(Path(train_csv_path))
    val_df = pd.read_csv(Path(val_csv_path))
    
    if train_y_csv_path is not None:
        train_y = pd.read_csv(Path(train_y_csv_path)).values
    else: 
        train_y = np.array([-1]*len(train_df)) 
    
    if val_y_csv_path is not None:
        val_y = pd.read_csv(Path(val_y_csv_path)).values
    else: 
        val_y = np.array([-1]*len(train_df))
    
    test_dataset = None
    if test_csv_path is not None:
        test_df = pd.read_csv(Path(test_csv_path))
        
        if test_y_csv_path is not None:
            test_y = pd.read_csv(Path(test_y_csv_path)).values
        else: 
            test_y = np.array([-1]*len(test_df))
        test_dataset = DatasetTabular(test_df.values, test_y)
    
    train_dataset = DatasetTabular(train_df.values, train_y)
    val_dataset = DatasetTabular(val_df.values, val_y)

    return train_dataset, val_dataset, test_dataset