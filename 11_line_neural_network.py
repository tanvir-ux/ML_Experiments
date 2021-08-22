import numpy as np;
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #input
y = np.array([[0,1,1,0]]).T #output label
w1 = 2*np.random.random((3,4)) - 1 #weights1
w2 = 2*np.random.random((4,1)) - 1 #weights2
for j in range(60000):
    layer1 = 1 / (1 + np.exp(-(np.dot(X,w1)))) #activate neuron
    layer2 = 1/(1 + np.exp(-(np.dot(layer1,w2)))) #activate neuron
    layer2_delta = (y - layer2) * (layer2 * (1 - layer2)) #backpropagate
    layer1_delta = layer2_delta.dot(w2.T) * (layer1 * (1 - layer1)) #backpropagate
    w2 += layer1.T.dot(layer2_delta) #update weights
    w1 += X.T.dot(layer1_delta) #update weights


# it was originally written by trask
# Eleven(11) line neural network    