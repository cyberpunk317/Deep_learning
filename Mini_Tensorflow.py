import numpy as np


class Node(object):
    """
    Base class for nodes in the network.

    Arguments:

        `inbound_nodes`: A list of nodes with edges into this node.
    """
    def __init__(self, inbound_nodes=[]):
        # A list of nodes with edges into this node.
        self.inbound_nodes = inbound_nodes
        # The eventual value of this node. Set by running
        # the forward() method.
        self.value = None
        # A list of nodes that this node outputs to.
        self.outbound_nodes = []
        self.gradients = {}
        # Sets this node as an outbound node for all of
        # this node's inputs.
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        """
        Every node that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Input(Node):
    """
    A generic input into the network.
    """
    def __init__(self):
        Node.__init__(self)

    def forward(self):
        pass

    def backward(self): 
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]

class Linear(Node):
    """
    Represents a node that performs a linear transform.
    """
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        """
        Performs the math behind a linear transform.
        """
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]   
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self, node):
        # The base class constructor.
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        # Sum the partial with respect to the input over all the outputs.
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost

class ReLU(Node):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self, node):
        Node.__init__(self, [node])

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._relu(input_value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            relu = self.value
            relu[relu>0] = 1
            self.gradients[self.inbound_nodes[0]] += relu * grad_cost

class LeakyReLU(Node):
    """
    Represents a node that performs the Leaky ReLU activation function.
    """
    def __init__(self, node):
        Node.__init__(self, [node])

    def _leaky_relu(self, x):
        temp = 0.01*x
        return np.maximum(temp, x)

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._leaky_relu(input_value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            leaky_relu = self.value
            leaky_relu[leaky_relu>0] = 1
            leaky_relu[leaky_relu<0] = 0.01
            self.gradients[self.inbound_nodes[0]] += leaky_relu * grad_cost


class MSE(Node):
    def __init__(self, y, a):
        """
        The mean squared error cost function.
        """
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff


def topological_sort(feed_dict):
    """
    Sort the nodes in topological order using Kahn's Algorithm.

    Returns a list of sorted nodes.
    """

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):
    """
    Performs a forward pass and a backward pass through a list of sorted Nodes.

    """
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    for n in graph[::-1]:
        n.backward()


def sgd_update(trainables, learning_rate=1e-2):
    """
    Updates the value of each trainable with SGD.

    """
    for trainable in trainables:
        trainable.value -= learning_rate*trainable.gradients[trainable]