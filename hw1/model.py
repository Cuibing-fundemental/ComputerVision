from utils import Fc,Relu,Module,Sigmoid

class MLP(Module):
    def __init__(self, hidden1 = 512, hidden2 = 256, nonlin = "relu"):
        super().__init__()
        self.fc1 = Fc(784,hidden1)
        self.fc2 = Fc(hidden1,hidden2)
        self.fc3 = Fc(hidden2,10)
        if nonlin == "sigmoid":
            self.nonlin = Sigmoid()
        else: 
            self.nonlin = Relu()

    def forward(self,x):
        x = self.fc1(x)
        x = self.nonlin(x)
        x = self.fc2(x)
        x = self.nonlin(x)
        x = self.fc3(x)
        return x
