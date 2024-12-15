
# 🕯️ **PyCandle 2: Ignite Your Understanding of Deep Learning**  

> **“Sometimes, the best way to learn how something works is to build it yourself.”**  

**PyCandle 2** is not just another deep learning library – it’s your gateway to understanding what happens *under the hood* of frameworks like PyTorch. Designed for students, researchers, and deep learning enthusiasts, PyCandle lets you dive deep into the mathematics  and magic of **neural networks**, **gradients**, and **computational graphs**.  

Built entirely with **NumPy** (and optional **CuPy** for GPU acceleration), PyCandle is minimal, scalable and most importantly – **educational**.  

> **✨ This is an improved and enhanced version of the original [PyCandle library](https://github.com/TimaGitHub/pycandle)**, now with full computational graph support, extended modules, and GPU acceleration!  

---

![pycandle2](https://github.com/user-attachments/assets/b51f765d-19cd-499f-91f0-07cec3acab93)
---

## 🔥 **Why PyCandle?**

### 🌟 **Learn from Scratch**  
- Build and experiment with neural networks, layer by layer.  
- Mimics the **PyTorch API** to keep learning smooth and intuitive.  

### ⚙️ **What You Get**  
- A **fully functional computational graph** that tracks gradients for backpropagation.  
- Support for **Linear**, **Convolutional**, **Embedding**, **BatchNorm**, **LayerNorm**, and more.  
- Key **activation functions**: ReLU, Sigmoid, Tanh, Leaky ReLU, Softmax, and GeLU.  
- Gradient-based **optimizers**: SGD, NAG, RMSProp, and Adam.  
- **Seamless GPU acceleration** using **CuPy**.  

### 🧠 **What's Under the Hood?**  
- Every operation, from matrix multiplications to activations, integrates smoothly into the computational graph.  
- Backpropagation is automatic and transparent – allowing you to see **how gradients flow** through your network.  

---

## 🚀 **Getting Started**  

### **Step 1: Build a Simple Neural Network**  

```python
import candle
import candle.nn as nn
from candle.utils.data import DataLoader
from candle.utils import accuracy
from candle.optim import SGD, ADAM, RMSProp, NAG
from candle import Tensor

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

# Instantiate model, loss, and optimizer
model = SimpleNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    logits = model(x)
    model.zero_grad()
    loss = loss_fn(logits, target)
    loss.backward()
    optimizer.step()
```

### **Step 2: Move to GPU in One Line**  
```python
model.to('gpu')  # Switch from CPU to GPU seamlessly
```

---

## 🎨 **A Peek at the PyCandle Ecosystem**  

### 🔹 **Core Tensor Class**  
At the heart of PyCandle is the **Tensor** – a flexible, gradient-aware structure that integrates seamlessly into the computational graph.  

#### Basic Operations  
```python
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6])
c = a + b  # Automatically builds the graph
gradients = c.backward()  # Computes gradients

print("Gradients:\n", gradients[a])

>>> Gradients:
>>> [1. 1. 1.]
```

#### Element-wise Operations  
The Tensor class supports arithmetic operations, activation functions, and more:  
```python
x = Tensor([[1, -2], [3, -4]], requires_grad=True)
y = Tensor.relu(x)  # Apply ReLU activation
z = y ** 2          # Element-wise square
gradients = z.backward()  # Computes gradients through the computational graph

print("Input Tensor:\n", x)
print("Gradients:\n", gradients[x])

>>> Input Tensor:
>>> array([[ 1., -2.],
       [ 3., -4.]])
>>> Gradients:
>>> [[2. 0.]
    [6. 0.]]
```

#### Matrix Operations  
Matrix multiplications and advanced tensor manipulations are also fully supported:  
```python
w = Tensor([[1, 2], [3, 4]], requires_grad=True)
v = Tensor([[2, 0], [1, 3]])

result = w @ v  # Matrix multiplication
gradients = result.backward()  # Backpropagation

print("Result of Matrix Multiplication:\n", result)
print("Gradient of w:\n", gradients[w])

>>> array([[ 4.,  6.],
       [10., 12.]])
>>> Gradient of w:
>>>  [[2. 4.]
    [2. 4.]]
```

#### Move to GPU for Faster Computations  
Switch seamlessly between CPU and GPU using **CuPy**:  
```python
a = Tensor([[6.0, 2.0], [-1.0, 4.0]], requires_grad=True).to('gpu')
b = Tensor([[2.0, 10.0], [1.0, 3.0]]).to('gpu')

c = a @ b  # GPU-based matrix multiplication
gradients = c.backward()

print("Result on GPU:\n", c)
print(c.device)

>>> array([[14., 66.],
       [ 2.,  2.]])
>>> gpu
```
---

### 🔹 **Supported Layers**  
- **Fully Connected**: `Linear`  
- **Convolutional**: `Conv`  
- **Normalization**: `BatchNorm`, `LayerNorm`  
- **Regularization**: `Dropout`  
- **Submodules**: `ModuleList`, `Embedding`  
- **Recurrent Networks**: Coming soon – `RNN`, `LSTM` and `GRU`!  

### 🔹 **Optimizers**  
Choose from classic optimization algorithms:  
- **SGD**: Stochastic Gradient Descent  
- **NAG**: Nesterov Accelerated Gradient
- **RMSProp**: Root Mean Square Propagation  
- **Adam**: Adaptive Momentum  

---

## 🧩 **Example: Building a Convolutional Network**  

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv(1, 8, (3, 3))  # Convolutional layer
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 26 * 26, 10)  # Fully connected

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.flatten(x)
        return self.fc1(x)
```

---

## 🚧 **To-Do List**  

- [ ] Add **RNN**, **LSTM**, **GRUs**.  
- [ ] Check for the correctness of the convolutional operations.  
- [ ] Check for the correctness of NAG, RMSProp, Adam.
- [ ] Provide computational graph for indexing, stack, split, cat, min, max etc... operations.
- [ ] Replace CuPy with Triton or Cuda.
- [ ] Replace NumPy with C++ library.
- [ ] Make Embdedding.forward() method more efficient as it is currently impossible to train transformers due to low perfomance of operation.
- [ ] Redesign library
---

## ✨ **PyCandle: Build It, Learn It, Master It**  

Start exploring the building blocks of deep learning today! Whether you're a student eager to understand neural networks or a researcher experimenting with custom implementations, PyCandle is your perfect companion.  

🔥 **Light up your learning with PyCandle!** 🔥  




