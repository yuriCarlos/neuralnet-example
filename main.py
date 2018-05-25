from net import Net


def xor_problem():

    net=Net(class_divisor=0.5)
    net.create_layer(3)
    net.create_layer(2)
    net.create_layer(1)
    print(net.layers)

    X = [
        
        [.9,.9,.9],
        [.9,.9,.1],
        [.9,.1,.9],
        [.9,.1,.1]
    ]

    Y=[
        [0],
        [1],
        [1],
        [0]
    ]

    net.train(X, Y, batch=1)
    print(net.predict(X))
    print(Y)