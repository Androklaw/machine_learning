import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif

def data_split(data_df, p=0.7, seed=42):
    np.random.seed(seed)
    train_idxs = data_df.sample(int(len(data_df)*(1-p))).index
    test_idxs = ~data_df.index.isin(train_idxs)
    train_df = data_df.loc[train_idxs].reset_index(drop=True)
    test_df = data_df.loc[test_idxs].reset_index(drop=True)
    return train_df, test_df


euclid_distance = lambda x,p: np.sqrt(((x - p) ** 2).sum(1)).rename('euclid_distance')
pairwise_distances = lambda A, B: np.sqrt((A**2).sum(axis=1, keepdims=True) + (B**2).sum(axis=1) - 2 * A @ B.T)

filepath = './Poker/poker-hand-training-true.data'
target_name = 'Class'

data_df = pd.read_csv(filepath)
input_features = [c for c in data_df.columns if c != target_name]

for test_p in tqdm(np.arange(10, 100, 10)/100, 'Running...'):
    train_df, test_df = data_split(data_df, p=test_p)
    
    
    train_info_gain = mutual_info_classif(train_df[input_features], train_df[target_name])

    distances = pairwise_distances(train_df[input_features].values, test_df[input_features].values).T
    # distaces = pairwise_distances(train_df[input_features], test_df[input_features])

    for k in range(1, 9, 2):
        winners_indices = np.argpartition(distances, k, axis=1)[:, :k]
        k_labels = train_df[target_name].values[winners_indices]

        #Simple vote
        simple_vote_class = pd.DataFrame(k_labels.T).value_counts().idxmax()
        test_df.loc[:, f'k{k}:simple_vote_class'] = simple_vote_class
        
        #Distance based
        k_distances = np.take_along_axis(distances, winners_indices, axis=1)
        k_weights = k_distances/k_distances.sum(axis=1, keepdims=True)
        distance_based_class = [np.bincount(k_labels[i], weights=k_weights[i]).argmax() for i in range(len(k_labels))]
        test_df.loc[:, f'k{k}:distance_based_class'] = distance_based_class
