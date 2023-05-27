import pandas as pd
import numpy as np
import math as m
import bisect
import random
import copy


class Node:
    def __init__(self):
        self.value = None
        self.decision = None
        self.children = []

def id3(examples, target_attribute, attributes, node, depth=5, actual_depth=0):
    classes = examples[target_attribute].unique()
    num_of_examples = len(examples)
    for class_value in classes:
        num_of_inner_examples = len(examples[examples[target_attribute] == class_value])
        if num_of_inner_examples == num_of_examples:
            node.value = class_value
            return
    if len(attributes) == 0 or actual_depth == depth:
        node.value = get_most_common_value(examples, target_attribute)
        return

    best_attribute = get_best_attribute(examples, target_attribute, attributes)
    node.value = best_attribute
    for example in examples[best_attribute].unique():
        new_node = Node()
        new_node.decision = example
        node.children.append(new_node)
        examples_vi = examples[examples[best_attribute] == example]
        if len(examples_vi) == 0:
            leaf = Node()
            new_node.children.append(leaf)
            leaf.value = get_most_common_value(examples, target_attribute)
        else:
            local_attributes = copy.copy(attributes)
            local_attributes.remove(best_attribute)
            id3(examples_vi, target_attribute, local_attributes, new_node, actual_depth=actual_depth+1)
    return

def get_best_attribute(data, target_attribute, attributes):
    max_inf_gain = -1
    best_attr = None
    for attribute in attributes:
        inf_gain = get_information_gain(data, target_attribute, attribute)
        if inf_gain > max_inf_gain:
            max_inf_gain = inf_gain
            best_attr = attribute
    return best_attr

def get_information_gain(data, target_attribute, attribute):
    sum_of_entropy = 0
    num_of_examples = len(data)
    attribute_values = data[attribute].unique()
    for attribute_value in attribute_values:
        attribute_value_examples = data[data[attribute] == attribute_value]
        count = len(attribute_value_examples)
        sum_of_entropy += count / num_of_examples * get_entropy(attribute_value_examples, target_attribute)
    return get_entropy(data, target_attribute) - sum_of_entropy

def get_entropy(data, target_attribute):
    class_list = data[target_attribute].unique()
    num_of_classes = len(class_list)
    num_of_examples = len(data)
    entropy = 0
    for class_value in class_list:
        inner_examples = data[target_attribute] == class_value
        num_of_inner_examples = len(data[inner_examples])
        if num_of_inner_examples == 0:
            continue
        elif num_of_inner_examples == num_of_examples:
            return 0
        inner_examples_proportion = num_of_inner_examples / num_of_examples
        entropy += inner_examples_proportion * m.log(inner_examples_proportion, num_of_classes)
    return -1 * entropy

def get_most_common_value(data, target_attribute):
    classes = data[target_attribute].unique()
    most_common_values = []
    max_num = 0
    for class_value in classes:
        num_of_inner_examples = len(data[data[target_attribute] == class_value])
        if num_of_inner_examples > max_num:
            max_num = num_of_inner_examples
            most_common_values = [class_value]
        elif num_of_inner_examples == max_num:
            most_common_values.append(class_value)
    if len(most_common_values) == 1:
        return most_common_values[0]
    return most_common_values[random.randint(0, len(most_common_values)-1)]

def accuracy(root, data, target_attribute):
    num_of_correct_answers = 0
    examples = len(data)
    for i, row in data.iterrows():
        node = root
        while len(node.children) != 0:
            temp = False
            for child in node.children:
                if child.decision == row[node.value]:
                    temp = True
                    node = child
                    break
            if not temp:
                break
        if node.value == row[target_attribute]:
            num_of_correct_answers += 1
    return num_of_correct_answers / examples

def discretization(data, target_attribute, attribute, splits=2):
    data = data.sort_values(attribute).reset_index(drop=True)
    boundary_ind, inf_gain = get_best_boundary(data, target_attribute)
    label = 1
    data.loc[:boundary_ind-1, attribute] = label
    data.loc[boundary_ind:, attribute] = label + 1
    label = 3
    boundaries = [boundary_ind]
    current_splits_count = 2
    while current_splits_count < splits:
        used_ind_of_ind, used_ind, new_ind, cur_inf_gain = -1, -1, -1, -1
        left_done, temp = False, False
        for ind_of_ind, ind in enumerate(boundaries):
            if ind_of_ind != 0:
                boundary_ind, inf_gain = get_best_boundary(data.loc[boundaries[ind_of_ind-1]:ind-1], target_attribute)
            else:
                boundary_ind, inf_gain = get_best_boundary(data.loc[:ind-1], target_attribute)
            if boundary_ind is not None:
                if inf_gain > cur_inf_gain:
                    temp = True
                    left_done = True
                    cur_inf_gain = inf_gain
                    new_ind = boundary_ind
                    used_ind = ind
                    used_ind_of_ind = ind_of_ind
            if ind_of_ind + 1 < len(boundaries):
                boundary_ind, inf_gain = get_best_boundary(data.loc[ind:boundaries[ind_of_ind+1]-1], target_attribute)
            else:
                boundary_ind, inf_gain = get_best_boundary(data.loc[ind:], target_attribute)
            if boundary_ind is not None:
                if inf_gain > cur_inf_gain:
                    temp = True
                    left_done = False
                    cur_inf_gain = inf_gain
                    new_ind = boundary_ind
                    used_ind = ind
                    used_ind_of_ind = ind_of_ind
        if not temp:
            continue
        if left_done:
            if used_ind_of_ind == 0:
                data.loc[:new_ind-1, attribute] = label
                data.loc[new_ind:used_ind-1, attribute] = label + 1
            else:
                data.loc[boundaries[used_ind_of_ind-1]:new_ind-1, attribute] = label
                data.loc[new_ind:used_ind-1, attribute] = label + 1
        else:
            if used_ind_of_ind + 1 == len(boundaries):
                data.loc[used_ind:new_ind-1, attribute] = label
                data.loc[new_ind:, attribute] = label + 1
            else:
                data.loc[used_ind:new_ind-1, attribute] = label
                data.loc[new_ind:boundaries[used_ind_of_ind+1]-1, attribute] = label + 1
        label *= 2
        bisect.insort_left(boundaries, new_ind)
        current_splits_count += 1
    data[attribute] = data[attribute].astype(int)
    return data

def get_best_boundary(data, target_attribute):
    _class, actual_ind, inf_gain = None, None, None
    for ind, row in data.iterrows():
        if not _class:
            _class = row[target_attribute]
            continue
        if row[target_attribute] != _class:
            current_inf_gain = get_boundary_information_gain(data, target_attribute, ind)
            if not inf_gain:
                inf_gain = current_inf_gain
                actual_ind = ind
            elif current_inf_gain > inf_gain:
                inf_gain = current_inf_gain
                actual_ind = ind
            _class = row[target_attribute]
    if not inf_gain:
        return None, 0
    return actual_ind, inf_gain

def get_boundary_information_gain(data, target_attribute, boundary_idx):
    examples = len(data)
    Dv_right = data.loc[:boundary_idx-1]
    Dv_left = data.loc[boundary_idx:]
    num_of_Dv_less = len(Dv_right)
    num_of_Dv_more = len(Dv_left)
    sum_of_entropy = num_of_Dv_less / examples * get_entropy(Dv_right, target_attribute)
    sum_of_entropy += num_of_Dv_more / examples * get_entropy(Dv_left, target_attribute)
    return get_entropy(data, target_attribute) - sum_of_entropy

def equal_discretization(data, attribute, groups):
    data = data.sort_values(attribute)
    column = data[attribute]
    col_min, col_max = column.min(), column.max()
    step = (col_max - col_min) / groups
    def get_discrete_value(x):
        n = col_min + step
        i = 0
        while n <= col_max + step:
            if x <= n:
                return i
            i += 1
            n += step
    data[attribute] = column.apply(lambda x: get_discrete_value(x)).astype(int)
    return data.sort_index()

def split_train_test(data, p=0.7):
    spliter = np.random.rand(len(data)) < p
    train = data[spliter]
    test = data[~spliter]
    return train, test

def print_tree(node, spaces=0):
    print('  ' * spaces, end='')
    if node.decision is None:
        print(node.value)
    else:
        print(f'{node.decision} - {node.value}')
    if not node.children:
        return
    else:
        for child in node.children:
            print_tree(child, spaces + 1)


iris_data = pd.read_csv('Iris/iris.data')

attributes = list(iris_data.columns.values)
target_attribute = 'Class'
attributes.remove(target_attribute)



equal_discretized_iris = iris_data.copy(deep=True)
for attr in attributes:
    equal_discretized_iris = equal_discretization(equal_discretized_iris, attr, 4)

root_iris_equal_discretized = Node()
iris_train, iris_test = split_train_test(equal_discretized_iris, 0.7)
id3(iris_train, target_attribute, attributes, root_iris_equal_discretized)
print_tree(root_iris_equal_discretized)
print(accuracy(root_iris_equal_discretized, iris_test, target_attribute))



ig_discretized_iris = iris_data.copy(deep=True)
for attr in attributes:
    ig_discretized_iris = discretization(ig_discretized_iris, target_attribute, attr, 4)

root_iris_ig_discretized = Node()
iris_train, iris_test = split_train_test(ig_discretized_iris, 0.7)
id3(iris_train, target_attribute, attributes, root_iris_ig_discretized)
print_tree(root_iris_ig_discretized)
print(accuracy(root_iris_ig_discretized, iris_test, target_attribute))



# poker_train = pd.read_csv('Poker/poker-hand-training-true.data')
# poker_test = pd.read_csv('Poker/poker-hand-testing.data')
# attributes = list(poker_train.columns.values)
# target_attribute = 'CLASS'
# attributes.remove(target_attribute)
#
#
#
# equal_discretized_poker_train = poker_train.copy(deep=True)
# equal_discretized_poker_test = poker_test.copy(deep=True)
# for attr in attributes:
#     equal_discretized_poker_train = equal_discretization(equal_discretized_poker_train, attr, 4)
#     equal_discretized_poker_test = equal_discretization(equal_discretized_poker_test, attr, 4)
#
# poker_root = Node()
# id3(equal_discretized_poker_train, target_attribute, attributes, poker_root)
# print_tree(poker_root)
# print(accuracy(poker_root, equal_discretized_poker_test, target_attribute))



# ig_discretized_poker_train = poker_train.copy(deep=True)
# ig_discretized_poker_test = poker_test.copy(deep=True)
# for attr in attributes:
#     ig_discretized_poker_train = discretization(ig_discretized_poker_train, target_attribute, attr, 4)
#     ig_discretized_poker_test = discretization(ig_discretized_poker_test, target_attribute, attr, 4)
#
# poker_root = Node()
# id3(ig_discretized_poker_train, target_attribute, attributes, poker_root)
# print_tree(poker_root)
# print(accuracy(poker_root, ig_discretized_poker_test, target_attribute))
