import mnist_loader
training_data, validation_data, testing_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

# import network 
# net = network.Network([784, 30, 10])

# net.SGD(training_data, 30, 10, 3, test_data=testing_data)

import network_1
net1 = network_1.Network([784,30,10])
net1.SGD(training_data,30,10,3.0,test_data = testing_data)