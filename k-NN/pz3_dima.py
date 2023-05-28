import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns

np.seterr(divide='ignore')

def data_split(data_df, p=0.7, seed=42):
    np.random.seed(seed)
    train_idxs = data_df.sample(int(len(data_df)*(1-p))).index
    test_idxs = ~data_df.index.isin(train_idxs)
    train_df = data_df.loc[train_idxs].reset_index(drop=True)
    test_df = data_df.loc[test_idxs].reset_index(drop=True)
    return train_df, test_df


euclid_distance = lambda x,p: np.sqrt(((x - p) ** 2).sum(1)).rename('euclid_distance')
pairwise_distances = lambda A, B: np.sqrt((A**2).sum(axis=1, keepdims=True) + (B**2).sum(axis=1) - 2 * A @ B.T)

filepath = './Iris/iris.data' #'./Poker/poker-hand-training-true.data'
target_name = 'Class'

data_df = pd.read_csv(filepath)
data_df = data_df.drop_duplicates()
input_features = [c for c in data_df.columns if c != target_name]

data_df[input_features] = (data_df[input_features] - data_df[input_features].min(0))/data_df[input_features].mean(0)
uniqs = data_df[target_name].unique()
data_df[f'{target_name}:int'] = data_df[target_name].replace(dict(zip(uniqs, range(len(uniqs)))))


all_accuracies = []
for test_p in tqdm(np.arange(10, 100, 10)/100, 'Running...'):
    train_df, test_df = data_split(data_df, p=test_p)

    #Calculate mutual info between train input and output
    train_info_gain = mutual_info_classif(train_df[input_features], train_df[f'{target_name}:int'])
    train_info_gain = train_info_gain[np.newaxis, :]/train_info_gain.sum()
    
    general_distances = pairwise_distances(
        train_df[input_features].values,
        test_df[input_features].values
    ).T

    attrbased_distances = pairwise_distances(
        train_df[input_features].values * train_info_gain,
        test_df[input_features].values * train_info_gain
    ).T

    accuracies = {}
    accuracies['test_p'] = test_p
    for k in range(1, 9, 2):
        for dname, distances in zip(['general', 'attrbased'], [general_distances, attrbased_distances]):
            
            winners_indices = np.argpartition(distances, k, axis=1)[:, :k]
            k_labels = train_df[f'{target_name}:int'].values[winners_indices]

            #Simple vote
            simple_vote_class = pd.DataFrame(k_labels.T).value_counts().idxmax()
            accuracies[f'k{k}:simple_vote_class:{dname}'] = np.mean(simple_vote_class == test_df[f'{target_name}:int'])

            #Distance based
            k_distances = np.take_along_axis(distances, winners_indices, axis=1)
            k_weights = k_distances/k_distances.sum(axis=1, keepdims=True)
            distance_based_class = [np.bincount(k_labels[i], weights=k_weights[i]).argmax() for i in range(len(k_labels))]
            accuracies[f'k{k}:distance_based_class:{dname}'] = np.mean(distance_based_class == test_df[f'{target_name}:int'])
    all_accuracies.append(accuracies)  
acc_df = pd.DataFrame(all_accuracies)
# acc_df.to_csv('accuracies.csv', index=False)

chart_type = 'simple_vote_class:general'
cols = [c for c in acc_df.columns if chart_type in c]

macc_df = acc_df.melt(id_vars='test_p', value_vars=cols, var_name='chart_type', value_name='accuracy')
macc_df['k'] = macc_df['chart_type'].apply(lambda x: x.split(':')[0])

sns.lineplot(data=macc_df, x='test_p', y='accuracy', hue='k')



