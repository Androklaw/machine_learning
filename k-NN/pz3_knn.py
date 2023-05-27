from knn_functions import *
import json
# import matplotlib
# # matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# simple, distance_weighted, attribute_weighted, combined = get_tables('../Iris/iris.data')
# json.dump({
#    'simple': simple,
#     'distance_weighted': distance_weighted,
#     'attribute_weighted': attribute_weighted,
#     'combined': combined
# }, open('iris_knn.json', 'w'))
# print_tables(simple, distance_weighted, attribute_weighted, combined)
#
# simple, distance_weighted, attribute_weighted, combined = get_tables('../Wine/wine.data')
# json.dump({
#    'simple': simple,
#     'distance_weighted': distance_weighted,
#     'attribute_weighted': attribute_weighted,
#     'combined': combined
# }, open('wine_knn.json', 'w'))
# print_tables(simple, distance_weighted, attribute_weighted, combined)

simple, distance_weighted, attribute_weighted, combined = get_tables('../Poker/poker-hand-training-true.data')
json.dump({
    'simple': simple,
    'distance_weighted': distance_weighted,
    'attribute_weighted': attribute_weighted,
    'combined': combined
}, open('poker_knn.json', 'w'))
print_tables(simple, distance_weighted, attribute_weighted, combined)


# data = json.load(open('iris_knn.json'))
# simple = np.asarray(data['simple'])
# distance_weighted = np.asarray(data['distance_weighted'])
# attribute_weighted = np.asarray(data['attribute_weighted'])
# combined = np.asarray(data['combined'])
# print_tables(simple, distance_weighted, attribute_weighted, combined)

# data = json.load(open('wine_knn.json'))
# simple = np.asarray(data['simple'])
# distance_weighted = np.asarray(data['distance_weighted'])
# attribute_weighted = np.asarray(data['attribute_weighted'])
# combined = np.asarray(data['combined'])
# print_tables(simple, distance_weighted, attribute_weighted, combined)
#
data = json.load(open('poker_knn.json'))
simple = np.asarray(data['simple'])
distance_weighted = np.asarray(data['distance_weighted'])
attribute_weighted = np.asarray(data['attribute_weighted'])
combined = np.asarray(data['combined'])
print_tables(simple, distance_weighted, attribute_weighted, combined)

linestyles = ['dotted', 'dashed', (5, (10, 3)), ':']

# data = json.load(open('iris_knn.json'))
# simple = np.asarray(data['simple'])
# distance_weighted = np.asarray(data['distance_weighted'])
# X = np.linspace(10, 90, 9, dtype=int)
# for i in range(len(simple[0])):
#     plt.plot(X, simple[:, i], label=f'k={i * 2 + 1}', linestyle=linestyles[i])
# plt.xlabel = 'Percentage of training sample'
# plt.ylabel = 'Accuracy'
# plt.title('Iris results for simple knn')
# plt.legend()
# plt.show()
#
# for i in range(len(distance_weighted[0])):
#     plt.plot(X, distance_weighted[:, i], label=f'k={i * 2 + 1}', linestyle=linestyles[i])
# plt.xlabel = 'Percentage of training sample'
# plt.ylabel = 'Accuracy'
# plt.title('Iris results for distance weighted knn')
# plt.legend()
# plt.show()

# data = json.load(open('wine_knn.json'))
# attribute_weighted = np.asarray(data['attribute_weighted'])
# X = np.linspace(10, 90, 9, dtype=int)
# for i in range(len(attribute_weighted[0])):
#     plt.plot(X, attribute_weighted[:, i], label=f'k={i * 2 + 1}', linestyle=linestyles[i])
# plt.xlabel = 'Percentage of training sample'
# plt.ylabel = 'Accuracy'
# plt.title('Wine results for attribute weighted knn')
# plt.legend()
# plt.show()
#
data = json.load(open('poker_knn.json'))
simple = np.asarray(data['simple'])
combined = np.asarray(data['combined'])
X = np.linspace(10, 90, 9, dtype=int)
for i in range(len(simple[0])):
    plt.plot(X, simple[:, i], label=f'k={i * 2 + 1}', linestyle=linestyles[i])
plt.xlabel = 'Percentage of training sample'
plt.ylabel = 'Accuracy'
plt.title('Poker results for simple knn')
plt.legend()
plt.show()

for i in range(len(combined[0])):
    plt.plot(X, combined[:, i], label=f'k={i * 2 + 1}', linestyle=linestyles[i])
plt.xlabel = 'Percentage of training sample'
plt.ylabel = 'Accuracy'
plt.title('Poker results for combined knn')
plt.legend()
plt.show()
