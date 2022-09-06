from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# load iris dataset
iris = datasets.load_iris()

# convert dataset to pandas DataFrame
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                  columns=iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
df.columns = ['s_length', 's_width', 'p_length', 'p_width', 'target', 'species']

train, test = train_test_split(df, test_size=0.15, train_size=0.85)

# appending by 1
ones = {'Ones': 1}
train = train.assign(**ones)
test = test.assign(**ones)

c1 = train[(train.target == 0)]
c2 = train[(train.target == 1)]
c3 = train[(train.target == 2)]


def calculateLDF(z):
    # generating b
    bt = []
    for i in range(len(z.Ones)):
        bt.append(1)
    b = np.array([bt])
    b = b.T

    # multiply zT and b
    zT = z.T
    zT1 = zT.to_numpy()
    ztb = np.matmul(zT1, b)

    # multiplying Z and Zt
    z11 = z.to_numpy()
    z1zT = np.matmul(zT, z11)

    # getting the inverse of the Z
    zINV = pd.DataFrame(np.linalg.pinv(z1zT.values), z1zT.columns, z1zT.index)

    w = np.matmul(zINV, ztb)
    return w


def predict(x):
    dx1 = np.matmul(x, w1)
    dx2 = np.matmul(x, w2)
    dx3 = np.matmul(x, w3)
    if dx1[0] > 0 and dx2[0] < 0 and dx3[0] < 0:
        return 0.0
    elif dx1[0] < 0 and dx2[0] > 0 and dx3[0] < 0:
        return 1.0
    elif dx1[0] < 0 and dx2[0] < 0 and dx3[0] > 0:
        return 2.0
    elif dx1[0] < 0 and dx2[0] < 0 and dx3[0] < 0:
        return "New Class"
    else:
        return "Undefined"


c1 = c1[['s_length', 's_width', 'p_length', 'p_width', 'Ones']]
z1 = c1.append(-1 * c2[['s_length', 's_width', 'p_length', 'p_width', 'Ones']], ignore_index=True)
z1 = z1.append(-1 * c3[['s_length', 's_width', 'p_length', 'p_width', 'Ones']], ignore_index=True)
w1 = calculateLDF(z1)

c2 = c2[['s_length', 's_width', 'p_length', 'p_width', 'Ones']]
z2 = c2.append(-1 * c3[['s_length', 's_width', 'p_length', 'p_width', 'Ones']], ignore_index=True)
z2 = z2.append(-1 * c1[['s_length', 's_width', 'p_length', 'p_width', 'Ones']], ignore_index=True)

w2 = calculateLDF(z2)

c3 = c3[['s_length', 's_width', 'p_length', 'p_width', 'Ones']]
z3 = c3.append(-1 * c1[['s_length', 's_width', 'p_length', 'p_width', 'Ones']], ignore_index=True)
z3 = c3.append(-1 * c2[['s_length', 's_width', 'p_length', 'p_width', 'Ones']], ignore_index=True)

w3 = calculateLDF(z3)

prediction = [6.7, 3.1 ,5.6, 2.4] # 2
prediction.append(1)
print(predict(prediction))


results = []
for index, row in test.iterrows():
    testing = [row['s_length'], row['s_width'], row['p_length'], row['p_width']]
    testing.append(1)
    result = predict(testing)
    results.append(result)
ctr = 0
targett = test[['target']].to_numpy().T
for i, j in zip(targett[0], results):
    if (i == j):
        print("Class:", j)
    if (i != j):
        ctr += 1
        print("Wrong Classify as class", j, "while it belongs to class", i)
accurate = 23 - ctr
Accuracy = accurate / 23
print("Accuracy:", Accuracy * 100, "%")
