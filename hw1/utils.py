import numpy as np
from typing import List,Iterable,Set
from collections import OrderedDict
import json

"""
basic class for value and gradient calculation
"""
class Value():
    def __init__(self, data:np.ndarray, prev: Iterable['Value'] = (),op = ""):
        data = data.astype(np.float64)
        if data.ndim == 1:
            self.data = data[None,:]
        else:
            self.data = data
        self._prev: Set['Value'] = set(prev)
        self.grad = np.zeros_like(self.data)
        self._op = op
        self._isvector = (data.ndim == 1)

    @property
    def shape(self):
        return self.data.shape
    
    def _backward(self):
        pass

    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        result = Value(self.data+other.data, (self,other),"+")

        def _backward():
            self.grad += result.grad
            if other.shape == result.shape:
                other.grad += result.grad
            else:
                other.grad += np.sum(result.grad, axis= 0, keepdims= True)
        result._backward = _backward

        return result

    def __matmul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        if self.shape[1] != other.shape[0]:
            raise ValueError("size not match")
        result = Value(self.data@other.data, (self,other),"@")

        def _backward():
            self.grad += result.grad @ other.data.T
            other.grad += self.data.T @ result.grad
        result._backward = _backward

        return result

    def __mul__(self, other:float):
        result = Value(self.data * other,(self,),"*")
        def _backward():
            self.grad += result.grad * other
        result._backward = _backward
        return result
    
    def __truediv__(self,other:float):
        result = self.__mul__(1./other)
        return result
    
    def __getitem__(self,key):
        return self.data[key]
    
    def backward(self):
        self.grad = np.ones_like(self.data)
        visit = set()
        topo = []
        def toposort(v):
            if v not in visit:
                visit.add(v)
                for child in v._prev:
                    toposort(child)
                topo.append(v)
        toposort(self)

        for v in reversed(topo):
            v._backward()

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

class Parameter(Value):
    pass

"""
Layer definition
"""
class Module():
    def __init__(self):
        self._parameters = OrderedDict()
        self._modules = OrderedDict()

    def zero_grad(self):
        for para in self.parameters():
            para.zero_grad()

    def __setattr__(self, name, value):
        if isinstance(value,Parameter):
            self._parameters[name] = value
        elif isinstance(value,Module):
            self._modules[name] = value
        else:
            super().__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._parameters:
            return self._parameters[name]
        elif name in self._modules:
            return self._modules[name]
        else:
            return self.__dict__[name]

    def parameters(self):
        for para in self._parameters.values():
            yield para
        for module in self._modules.values():
            for para in module.parameters():
                yield para

    def named_parameters(self, prefix = ''):
        for name, para in self._parameters.items():
            yield prefix + '.' + name, para
        for name, module in self._modules.items():
            for sub_name, para in module.named_parameters(prefix + '.' + name):
                yield sub_name, para

    def state_dict(self) -> OrderedDict:
        snapshot = OrderedDict()
        for name, para in self.named_parameters():
            snapshot[name] = para.data.copy().tolist()

        return snapshot
    
    def load(self, state_dict: OrderedDict):
        for name, para in self.named_parameters():
            para.data = np.array(state_dict[name])
    
    def update(self,lr):
        for para in self.parameters():
            para.data -= para.grad * lr

class Fc(Module):
    def __init__(self,indim,outdim,bias = True):
        super().__init__()
        self.weight = Parameter(np.random.normal(0,2/indim,(indim, outdim)),())
        self.bias_on = bias
        if self.bias_on:
            self.bias = Parameter(np.zeros(outdim),())

    def __call__(self,x):
        if self.bias_on:
            out = x @ self.weight + self.bias
        else:
            out = x @ self.weight
        return out


def relu(x:Value) -> Value:
    out = Value(np.maximum(0,x.data),(x,),"relu")

    def _backward():
        x.grad[x.data > 0] += out.grad[x.data > 0]
    out._backward = _backward

    return out

class Relu(Module):
    def __init__(self):
        super().__init__()

    def __call__(self,x):
        return relu(x)

def sigmoid(x:Value) -> Value:
    data = 1./(1.+np.exp(-x.data))
    out = Value(data, (x,), "sigmoid")

    def _backward():
        x.grad += out.data * (1-out.data) * out.grad
    out._backward = _backward

    return out

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def __call__(self,x):
        return sigmoid(x)


"""
loss function
"""
def cross_entropy(y_pred: Value, target_indices: list) -> Value:
    def softmax(y:Value) -> Value:
        max_val = np.max(y.data, axis=1, keepdims=True)
        nomin = np.exp(y.data - max_val)
        deno = np.sum(nomin, axis=1, keepdims=True)
        out = nomin/deno
        return out
    
    y_prob = softmax(y_pred)
    batch_size = y_prob.shape[0]
    row_indices = np.arange(batch_size)
    prob_val = y_prob[row_indices, target_indices]
    
    safe_prob = np.clip(prob_val, 1e-15, 1.0)
    loss_val = -np.log(safe_prob)
    total_loss = np.sum(loss_val)/batch_size
    
    out = Value(total_loss, (y_pred,), "CE")

    def _backward():
        grad = y_prob.copy()
        grad[row_indices, target_indices] -= 1.0
        y_pred.grad += out.grad* grad
    out._backward = _backward
    return out

def weight_decay(model:Module,lamb:float) -> Value:
    para_sum = 0
    for para in model.parameters:
        pass
"""
Evaluation
"""
def accuracy(y_pred: Value, target_indices: list) -> float:
    max_prob = np.argmax(y_pred.data, axis= 1)
    return np.average(max_prob == target_indices)

"""
Helper function
"""
def save_model(state_dict:OrderedDict,path):
    with open(path, 'w') as f:
        json.dump(state_dict, f, indent=4)

def load_model(path):
    with open(path, 'r') as f:
        state_dict = json.load(f, object_pairs_hook=OrderedDict)
    return state_dict

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels