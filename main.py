from net import Net
from sklearn import datasets

import numpy as np

def metrics(Y, p_Y):
    overall = 0
    for i in range(0, len(Y)):
        if Y[i][0] == p_Y[i][0]:
            overall += 1
    
    print(overall/len(Y))

def shuffle_data(X,Y):
    indexes = np.random.permutation(len(X))

    return [X[i] for i in indexes], [Y[i] for i in indexes]

def xor_problem():

    net=Net(class_divisor=0.5)
    net.create_layer(3)
    net.create_layer(2)
    net.create_layer(1, last_layer=True)
    print(net.layers)

    X = [
        
        [.99,.99,.99],
        [.99,.99,.1],
        [.99,.1,.99],
        [.99,.1,.1]
    ]

    Y=[
        [0],
        [1],
        [1],
        [0]
    ]

    net.train(X, Y, batch=1)
    res = net.predict(X)
    print(res)
    print(Y)
    metrics(Y,res)

xor_problem()

def normalize(min, max, x):
    return (x - min)/ (max-min)

def iris_problem():
    iris = datasets.load_iris()
    
    Y = []
    # deixa o target de acordo com a necessidade da rede
    for i in iris.target:
        Y.append([i])

    # pega os máximos e mínimos para normalização
    max_x = []
    min_x = []
    for column in range(0, len(iris.data[0])):
        max_x.append(max([i[column] for i in iris.data]))    
        min_x.append(min([i[column] for i in iris.data]))

    # faz um shuffle nos dados
    data, Y = shuffle_data(iris.data, Y)

    
    # seleciona as características e normaliza elas
    columns = [0, 2]
    X = []
    for x in data:
        X.append([normalize(min_x[column], max_x[column], x[column]) for column in columns])
    # data = aux
    print(X)

    # pega 100 primeiros dados para treino
    data_num=100
    x_test = X[:data_num]
    y_test = Y[:data_num]
    print(y_test)

    #cria a rede
    net = Net(class_divisor=1/3)
    net.create_layer(len(x_test[0]))
    net.create_layer(4)
    net.create_layer(1, last_layer=True)

    net.train(x_test, y_test, learning_rate=0.3, epoch=5000)

    res = net.predict(x_test)
    print(res)
    print(y_test)
    metrics(y_test,res)

# iris_problem()
