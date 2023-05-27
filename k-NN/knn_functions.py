import pandas as pd
import numpy as np
import time

def normalize(data, attributes):
    for attr in attributes:
        column = data[attr]
        mini, maxi = column.min(), column.max()
        data[attr] = column.apply(lambda x: (x - mini) / maxi)
    return data


def get_distance(x, y, attributes):
    dist = 0
    for attr in attributes:
        dist += (x[attr] - y[attr]) ** 2
        # if x[attr] == y[attr]:
        #     temp = 0
        # else:
        #     temp = 1
        # dist += temp
    return np.sqrt(dist)

def get_sorted_distances(x, train, attributes, k):

    distances = np.empty(train.shape[0])
    for idx, y in train.iterrows():
        distances[idx] = get_distance(x, y, attributes)
    distances = pd.DataFrame({'Distance': distances})
    distances = distances.sort_values(by=['Distance'])
    return distances
    # _distances = []
    # idx_distances = []
    # for i in range(k):
    #     _min = distances['Distance'].min()
    #     idx_min = distances['Distance'].idxmin()
    #     idx_distances.append(idx_min)
    #     _distances.append(_min)
    #     distances.drop(idx_min)
    # return pd.DataFrame({'Distance': _distances, 'index': idx_distances}).set_index('index')


def simple_voting(k_nearest, distances, target_attribute):
    value_counts = k_nearest[target_attribute].value_counts()
    return value_counts.idxmax()


def distance_weighted_voting(k_nearest, distances, target_attribute):
    unique_classes = {k: 0 for k in k_nearest[target_attribute].unique()}
    target_values = np.asarray(k_nearest[target_attribute])
    distances = np.asarray(distances['Distance'])
    for i in range(len(distances)):
        unique_classes[target_values[i]] += 1 / (distances[i] ** 2)
    common_class = max(unique_classes, key=unique_classes.get)
    return common_class


def knn(test, train, ks, attributes, target_attribute, voting_function):
    # test = normalize(test, attributes)
    # train = normalize(train, attributes)

    classes = [np.empty(test.shape[0], dtype='object') for _ in range(len(ks))]

    correct_answers = [0 for _ in range(len(ks))]
    for idx, x in test.iterrows():
        break
        distances = get_sorted_distances(x, train, attributes, 0)
        for i, k in enumerate(ks):
            # k_nearest_distances = get_sorted_distances(x, train, attributes, k)
            k_nearest_distances = distances.head(k)
            classes[i][idx] = voting_function(train.iloc[k_nearest_distances.index], k_nearest_distances,
                                              target_attribute)
            if classes[i][idx] == x[target_attribute]:
                correct_answers[i] += 1
    correct_answers = [c_a / len(test) for c_a in correct_answers]
    return classes, correct_answers


def calculate_entropy(data, target_attribute):
    _, counts = np.unique(data[target_attribute], return_counts=True)
    probabilities = counts / len(data[target_attribute])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def calculate_information_gain(data, target_attribute, attr):
    sum_of_entr = 0
    examples = len(data)
    values = data[attr].unique()
    for value in values:
        local_examples = data[data[attr] == value]
        count = len(local_examples)
        sum_of_entr += count / examples * calculate_entropy(local_examples, target_attribute)
    return calculate_entropy(data, target_attribute) - sum_of_entr


def attribute_weighted_knn(test, train, ks, attributes, target_attribute, combined=False):
    def get_weighted_distances(x, y, attributes, weights):
        dist = 0
        for i, attr in enumerate(attributes):
            dist += weights[i] * (x[attr] - y[attr]) ** 2
        return np.sqrt(dist)

    def get_sorted_distances(x, train, attributes, weights, k):
        distances = np.empty(train.shape[0])
        for idx, y in train.iterrows():
            distances[idx] = get_weighted_distances(x, y, attributes, weights)
        distances = pd.DataFrame({'Distance': distances})
        distances = distances.sort_values(by=['Distance'])
        return distances
        # _distances = []
        # idx_distances = []
        # for i in range(k):
        #     _min = distances['Distance'].min()
        #     idx_min = distances['Distance'].idxmin()
        #     idx_distances.append(idx_min)
        #     _distances.append(_min)
        #     distances.drop(idx_min)
        # return pd.DataFrame({'Distance': _distances, 'index': idx_distances}).set_index('index')

    # test = normalize(test, attributes)
    # train = normalize(train, attributes)

    classes = [np.empty(test.shape[0], dtype='object') for _ in range(len(ks))]

    correct_answers = [0 for _ in range(len(ks))]

    weights = []
    for attr in attributes:
        information_gain = calculate_information_gain(train, target_attribute, attr)
        weights.append(information_gain)
    for idx, x in test.iterrows():
        distances = get_sorted_distances(x, train, attributes, weights, 0)
        for i, k in enumerate(ks):
            k_nearest_distances = distances.head(k)
            # k_nearest_distances = get_sorted_distances(x, train, attributes, weights, k)
            if not combined:
                classes[i][idx] = simple_voting(train.iloc[k_nearest_distances.index], k_nearest_distances, target_attribute)
            else:
                classes[i][idx] = distance_weighted_voting(train.iloc[k_nearest_distances.index], k_nearest_distances, target_attribute)
            if classes[i][idx] == x[target_attribute]:
                correct_answers[i] += 1

    correct_answers = [c_a / len(test) for c_a in correct_answers]
    return classes, correct_answers


def split_train_test(data, p=0.7):
    np.random.seed(42)
    divider = np.random.rand(len(data)) < 0.3
    _data = data[divider]
    divider = np.random.rand(len(_data)) < p
    train = _data[divider]
    test = _data[~divider]
    return train, test


res_time = time.time()


def get_tables(file):
    data = pd.read_csv(file)
    attrs = data.columns.values
    class_attribute = 'Class'
    attrs = np.delete(attrs, np.where(attrs == class_attribute))

    train_data, test_data = split_train_test(data)
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    simple = []
    distance_wighted = []
    attribute_weighted = []
    combined = []
    print(f'file {file}')
    for proportion in range(10, 100, 10):
        print(f'\nproportion: {proportion}')
        proportion /= 100
        divider = np.random.rand(len(train_data)) < proportion
        part_of_train_data = train_data[divider].reset_index(drop=True)
        ks = list(range(1, 9, 2))
        start = time.time()
        _, accuracy = knn(test_data, part_of_train_data, ks, attrs, class_attribute, simple_voting)
        simple.append(accuracy)
        _, accuracy = knn(test_data, part_of_train_data, ks, attrs, class_attribute, distance_weighted_voting)
        distance_wighted.append(accuracy)
        _, accuracy = attribute_weighted_knn(test_data, part_of_train_data, ks, attrs, class_attribute)
        attribute_weighted.append(accuracy)
        _, accuracy = attribute_weighted_knn(test_data, part_of_train_data, ks, attrs, class_attribute, True)
        combined.append(accuracy)
        print(f'Time: {time.time() - start}')
    return simple, distance_wighted, attribute_weighted, combined


def print_tables(simple, distance_weighted, attribute_weighted, combined):
    # print(f'Result time: {time.time() - res_time}')
    print('Simple knn')
    print(f'{"Proportion/k": >12}', end='\t')
    for k in range(1, 9, 2):
        print(f'{k: >12}', end='\t')
    print()
    for proportion in range(10, 100, 10):
        print(f'{proportion: >12}', end='\t')
        for i in range(len(simple[0])):
            print(f'{simple[proportion // 10 - 1][i]:>12.3f}', end='\t')
        print()

    print('Distance weighted knn')
    print(f'{"Proportion/k": >12}', end='\t')
    for k in range(1, 9, 2):
        print(f'{k: >12}', end='\t')
    print()
    for proportion in range(10, 100, 10):
        print(f'{proportion: >12}', end='\t')
        for i in range(len(distance_weighted[0])):
            print(f'{distance_weighted[proportion // 10 - 1][i]:>12.3f}', end='\t')
        print()

    print('Attribute weighted knn')
    print(f'{"Proportion/k": >12}', end='\t')
    for k in range(1, 9, 2):
        print(f'{k: >12}', end='\t')
    print()
    for proportion in range(10, 100, 10):
        print(f'{proportion: >12}', end='\t')
        for i in range(len(attribute_weighted[0])):
            print(f'{attribute_weighted[proportion // 10 - 1][i]:>12.3f}', end='\t')
        print()

    print('Combined knn')
    print(f'{"Proportion/k": >12}', end='\t')
    for k in range(1, 9, 2):
        print(f'{k: >12}', end='\t')
    print()
    for proportion in range(10, 100, 10):
        print(f'{proportion: >12}', end='\t')
        for i in range(len(combined[0])):
            print(f'{combined[proportion // 10 - 1][i]:>12.3f}', end='\t')
        print()