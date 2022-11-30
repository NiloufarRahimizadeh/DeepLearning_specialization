import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def rnn_step_foward(parameters, a_prev, x):
    Wax, Waa, Wya, by, b = parameters["Wax"], parameters["Waa"], parameters["Wya"], parameters["by"], parameters["b"]
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
    p_t = softmax(np.dot(Wya, a_next) + by)
    
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dwya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters["Wya"].T, dy) + gradients["da_next"]
    daraw = (1 - a * a) * da
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] += np.dot(parameters['Waa'].T, daraw)
    
    return gradients


def update_parameters(parameters, gradients, lr):
    
    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b'] += -lr * gradients['db']
    parameters['by'] += -lr *  gradients['dby']
    
    return parameters


def rnn_forward(X, Y, a0, parameters, vocab_size=27):
    
    x, a, y_hat = {}, {}, {}
    a[-1] = np.copy(a0)
    loss = 0
    
    for t in range(len(X)):
        x[t] = np.zeros((vocab_size, 1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    