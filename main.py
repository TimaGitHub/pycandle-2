import candle
import candle.nn as nn
from candle.utils.data import DataLoader
from candle.utils import accuracy
from candle.optim import SGD, ADAM, RMSProp, NAG
from candle import Tensor

class Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 784, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        return x

class Temp2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear2 = Temp()

    def forward(self, x):
        x = self.linear2(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear3 = nn.ModuleList([Temp2() for i in range(5)])
        self.lein = nn.Linear(784, 20, bias=False)
        self.act = nn.ReLU()
        self.linear4 = nn.Linear(20, 10, bias=True)
        self.bn = nn.BatchNorm(10)

    def forward(self, x):
        x = self.linear3(x)
        x = self.lein(x)
        x = self.act(x)
        x = self.linear4(x)
        x = self.bn(x)
        return x


x = Tensor.randn((2, 784))
target = Tensor([[1],[2]])

y = Tensor((Tensor.arange(0, 10) == target) * 1)


model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
optim = SGD(model, lr=2e-4)

if __name__ == '__main__':
    output = model(x)
    model.zero_grad()
    loss = loss_fn(output, target)
    loss.backward()
    optim.step()











