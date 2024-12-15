import numpy
try:
    import cupy
except:
    pass
np = numpy

from candle.parameter import ParameterObj
from candle import Tensor

Parameter = None

class Module:

    def __init__(self):
        self._constructor_Parameter = ParameterObj()
        self.device = 'cpu'
        global Parameter
        Parameter = self._constructor_Parameter

    def train(self):
        for layer in self._constructor_Parameter.layers:
            if type(layer).__name__ == 'BatchNorm':
                layer.training = True
                layer.evaluating = False

    def eval(self):
        for layer in self._constructor_Parameter.layers:
            if type(layer).__name__ == 'BatchNorm':
                layer.training = True
                layer.evaluating = False

    def forward(self):
        pass

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            layers = value._constructor_Parameter.layers
            dicts = value._constructor_Parameter.calling

            global Parameter
            Parameter = self._constructor_Parameter
            value._constructor_Parameter = self._constructor_Parameter

            self._constructor_Parameter.layers += layers
            self._constructor_Parameter.calling.update(dicts)

        elif type(value).__name__ in ('Linear', 'Conv', 'RNN', 'BatchNorm', 'LayerNorm', 'Embedding', 'Dropout'):
            value.param = self._constructor_Parameter
            value.set_myself()

        elif type(value).__name__ in 'ModuleList':
            for index, layer in enumerate(value.layers):
                if type(value).__name__ in ('Linear', 'Conv', 'RNN', 'BatchNorm', 'LayerNorm', 'Embedding', 'Dropout'):
                    layer.param = self._constructor_Parameter
                    layer.set_myself()
                else:
                    self.__setattr__(f"{name}_{index}", layer)
        self.__dict__[name] = value

    def __call__(self, *args):
        global Parameter
        Parameter = self._constructor_Parameter
        return self.forward(*args)

    def to(self, device):
        global np
        if device == 'gpu':
            np = cupy
            self.device = 'gpu'
        else:
            np = numpy
            self.device = 'cpu'
        for index, layer in enumerate(self._constructor_Parameter.layers):
            if type(layer).__name__ in ('Linear', 'Conv'):
                layer.w.to(device)
                layer.b.to(device)
            elif type(layer).__name__ in ('BatchNorm', 'LayerNorm'):
                layer.gamma.to(device)
                layer.beta.to(device)

            elif type(layer).__name__ == 'Embedding':
                layer.w.to(device)

        return self

    def parameters(self):
        for layer in self._constructor_Parameter.layers:
            if type(layer).__name__ == 'Linear':
                print('Linear:', "weights:", self._constructor_Parameter.calling[layer][0].shape, ", bias:", self._constructor_Parameter.calling[layer][1].shape)

            elif type(layer).__name__ in ('ReLU', 'Leaky_ReLU', 'Sigmoid', 'Tanh', 'Flatten', 'GeLU'):
                print(type(layer).__name__)

            elif type(layer).__name__ == 'Conv':
                print("Conv:", self._constructor_Parameter.calling[layer][0][0].shape, ", {} kernels ,".format(len(self._constructor_Parameter.calling[layer][0])), "bias:", self._constructor_Parameter.calling[layer][1].shape)

            elif type(layer).__name__ == 'BatchNorm':
                print("BatchNorm:", "gamma:", self._constructor_Parameter.calling[layer][0].shape, "beta:", self._constructor_Parameter.calling[layer][1].shape)

            elif type(layer).__name__ == 'LayerNorm':
                print("LayerNorm:", "gamma:", self._constructor_Parameter.calling[layer][0].shape, "beta:", self._constructor_Parameter.calling[layer][1].shape)

            elif type(layer).__name__ == 'Embedding':
                print('Embedding:', "weights:", self._constructor_Parameter.calling[layer][0].shape)

    def zero_grad(self):
        for index, layer in enumerate(self._constructor_Parameter.layers):
            if type(layer).__name__ in ('Linear', 'Conv'):
                layer.w.grad = 0
                layer.b.grad = 0
            if type(layer).__name__ in ('BatchNorm', 'LayerNorm'):
                layer.gamma.grad = 0
                layer.beta.grad = 0

    def save(self, path='model_params', format='tar'):

        if format == 'tar':
            path=path + '.npy.tar'
        elif format == 'npy':
            path=path + '.npy'
        elif format == 'zip':
            path=path + '.npz'
        else:
            raise ValueError('Incorect format')

        self.to('cpu')
        if format in ('tar', 'npy'):
            with open(path, 'wb') as f:
                for layer in self._constructor_Parameter.layers:
                    if type(layer).__name__ in ('Linear', 'Conv', 'RNN', 'BatchNorm', 'LayerNorm', 'Embedding'):
                        numpy.save(f, layer.all_layers)
        else:
            all_layers = []
            for layer in self._constructor_Parameter.layers:
                if type(layer).__name__ in ('Linear', 'Conv', 'RNN', 'BatchNorm', 'LayerNorm', 'Embedding'):
                    all_layers.append(layer.all_layers)
            numpy.savez_compressed(path, *all_layers)
        print(f'Model saved -> {path}')

    def load(self, path='model_params.npy.tar'):
        self.to('cpu')
        if path[-3:] in ('tar', 'npy'):
            with open(path, 'rb') as f:
                for layer in self._constructor_Parameter.layers:
                    if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm', 'LayerNorm'):
                        tmp = numpy.load(f, allow_pickle=True)
                        layer.all_layers[0].value = tmp[0].value
                        layer.all_layers[1].value = tmp[1].value
                    elif type(layer).__name__ == 'Embedding':
                        tmp = numpy.load(f, allow_pickle=True)
                        layer.all_layers[0].value = tmp[0].value
        else:
            npz = numpy.load(path, allow_pickle=True)
            index = 0
            for layer in self._constructor_Parameter.layers:
                if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm', 'LayerNorm'):
                    tmp = npz[f'arr_{index}']
                    layer.all_layers[0].value = tmp[0].value
                    layer.all_layers[1].value = tmp[1].value
                    index += 1
                elif type(layer).__name__ == 'Embedding':
                    tmp = npz[f'arr_{index}']
                    layer.all_layers[0].value = tmp[0].value
                    index += 1
        print(f'Model downloaded')

class Linear:
    def __init__(self, input_channels: int, output_channels: int, bias = True):
        self.grad = None
        self.param = None

        if (isinstance(input_channels, int) & isinstance(output_channels, int)) == False:
            raise Exception("Incorrect linear layer initialization")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bias = bias
        self.w = Tensor.uniform(-0.01, 0.01, size=(self.input_channels, self.output_channels)) / np.sqrt(input_channels)
        self.w.local_gradients = None
        self.w.requires_grad = True
        if bias:
            self.b = Tensor.uniform(- 0.01, 0.01, size=self.output_channels)
            self.b.requires_grad = True
        else:
            self.b = Tensor.zeros(self.output_channels)

        self.all_layers = [self.w, self.b]

        #Parameter([self, self.w, self.b])

    def set_myself(self):
        self.param([self, self.w, self.b])


    def __call__(self, x):
        #result = x @ Parameter.calling[self][0] + Parameter.calling[self][1]
        if self.param:
            global Parameter
            Parameter = self.param

        result = x @ self.w + self.b
        return result

class Conv:

    def __init__(self, input_channels: int, output_channels: int, kernel_size: tuple, bias = True):
        if (isinstance(input_channels, int) & isinstance(output_channels, int) & isinstance(kernel_size, (tuple, list)) & (len(kernel_size) == 2)) == False:
            raise Exception("Incorrect convolution layer initialization")
        self.param = None
        self.bias = bias
        self.input_channels = input_channels
        self.kernel_size = (input_channels, kernel_size[0], kernel_size[1])
        self.n_filters = output_channels

        self.w = Tensor.randn((self.n_filters, input_channels, kernel_size[0], kernel_size[1]), ) * 1e-2
        self.w.local_gradients = None
        self.w.requires_grad = True

        self.b = Tensor.randn((self.n_filters, 1, 1))
        if self.bias:
            self.b.requires_grad=True

        self.all_layers = [self.w, self.b]

        #Parameter([self, self.w, self.b])

    def set_myself(self):
        self.param([self, self.w, self.b])


    def __call__(self, x):
        if self.param:
            global Parameter
            Parameter = self.param

        if x.ndim == 3 and x.shape[0] == 1:
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        elif x.ndim == 3 and (not x.shape[0] == 1):
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        elif x.ndim == 2 and x.shape[0] == x.shape[1]:
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
        elif x.ndim == 2 and x.shape[0] != x.shape[1]:
            x = x.reshape(x.shape[0], 1, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])))
        elif x.ndim > 4 or x.ndim == 1:
            raise Exception("Something wrong with input data into convolution layer")

        matrix = x
        kernel = self.w
        num_images, matrix_z, matrix_y, matrix_x = matrix.shape
        num_kernels, kernel_z, kernel_y, kernel_x = kernel.shape
        result_z, result_x, result_y = num_kernels, matrix_x - kernel_x + 1, matrix_y - kernel_y + 1
        new_matrix = Tensor.sliding_window_view(matrix, kernel_z, kernel_y, kernel_x)

        outz = new_matrix.shape[1]
        outy = new_matrix.shape[2]
        outx = new_matrix.shape[3]
        result = new_matrix.reshape(num_images * outx * outy, kernel_z * kernel_y * kernel_x) @ kernel.reshape(-1, num_kernels)
        return result.reshape(num_images, result_z, result_y, result_x) + self.b

class BatchNorm:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.param = None
        self.grad = None
        self.gamma = Tensor.ones(dim, requires_grad=True)
        self.beta = Tensor.zeros(dim, requires_grad=True)
        self.eps = eps
        self.momentum = momentum
        self.running_mean = 0
        self.running_std = 0
        self.training = True
        self.evaluating = False
        self.all_layers = [self.gamma, self.beta]

    def set_myself(self):
        self.param([self, self.gamma, self.beta])

    def __call__(self, x):
        if self.param:
            global Parameter
            Parameter = self.param
        if self.training:
            xmean = Tensor.mean(x, axis=0, keepdims=True)
            xstd = Tensor.std(x, axis=0, keepdims=True)
        else:
            xmean = self.running_mean
            xstd = self.running_std

        x = (x - xmean) / (xstd + self.eps)
        if self.training:
            self.running_mean = xmean * self.momentum + (1 - self.momentum) * self.running_mean
            self.running_std = xstd * self.momentum + (1 - self.momentum) * self.running_std
            self.running_mean.local_gradients = None
            self.running_std.local_gradients = None

        return self.gamma * x + self.beta

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.param = None
        self.gamma = Tensor.ones(dim, requires_grad=True)
        self.beta = Tensor.zeros(dim, requires_grad=True)
        self.eps = eps
        self.all_layers = [self.gamma, self.beta]
        self.grad = None

    def set_myself(self):
        self.param([self, self.gamma, self.beta])

    def __call__(self, x):
        if self.param:
            global Parameter
            Parameter = self.param
        xmean = Tensor.mean(x, axis=2, keepdims=True)
        xstd = Tensor.std(x, axis=2, keepdims=True)
        x = (x - xmean) / (xstd + self.eps)
        return self.gamma * x + self.beta

class Embedding:

    def __init__(self, num_emb, emb_dim):
        from scipy.sparse import csr_matrix
        self.w = Tensor.randn((num_emb, emb_dim), requires_grad=True)
        self.w.local_gradients = None
        self.num_embd = num_emb
        self.emb_dim = emb_dim
        self.param = None
        self.grad = None
        self.all_layers = [self.w]

    def __call__(self, x):

        if self.param:
            global Parameter
            Parameter = self.param
        # This implementation is very inefficient but very simple for backprop calculation
        # one_hot = Tensor(np.eye(self.num_embd, dtype= np.int8)[x.value])
        # return one_hot @ self.w
        def multiply_by_locgrad(path_value):
            temp = np.zeros_like(self.w.value)
            np.add.at(np.zeros_like(self.w.value), x.value, path_value)
            return temp
        x.value = x.value.astype(int)
        local_gradients = (('embd', self.w, multiply_by_locgrad),)
        return Tensor(self.w.value[x.value], local_gradients=local_gradients)

    def set_myself(self):
        self.param([self, self.w])

class Dropout:

    def __init__(self, q):
        self.param = None
        if q < 0 or q > 1:
            raise Exception("Incorrect probability value")
        self.q = q

    def __call__(self, x):
        if self.param:
            global Parameter
            Parameter = self.param
        mask = Tensor(np.random.choice([0, 1], x.value[0,:].shape, p=[self.q, 1 - self.q]), requires_grad=False)
        return x * mask / self.q

    def set_myself(self):
        self.param([self, self.w])

# class RNN:
#     '''
#     note:
#     E - input's dimension of one sample ( vector of features ) E - from "Embedding"
#     H - input's dimension of the vector of Hidden state
#     N - output's dimension of the vector
#     B - batch size
#     T - the length of the sequence (number of time periods (seconds) )
#     more here: https://qudata.com/ml/ru/NN_RNN_Torch.html#LSTM
#     '''
#
#     def __init__(self, input_size: int, hidden_size: int, N: int, n_layers=1, nonlinearity='tanh',  bias=True,):
#
#         if (isinstance(E, int) & isinstance(H, int) & isinstance(N, int)) == False:
#             raise Exception("Incorrect reccurent layer initialization")
#
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.bias = bias
#         self.nonlinearity = Tensor.tanh if nonlinearity == 'tanh' else Tensor.relu
#         self.n_layers = n_layers
#
#         self.vh = [Tensor.uniform(-0.01, 0.01, size=(self.input_size, self.hidden_size), requires_grad=True) for i in range(self.n_layers)]
#         self.bh = [Tensor.uniform(-0.01, 0.01, size=(self.hidden_size), requires_grad=True) for i in range(self.n_layers)]
#         self.wh = [Tensor.uniform(-0.01, 0.01, size=(self.hidden_size, self.hidden_size), requires_grad=True) for i in range(self.n_layers)]
#
#         self.layers = [self.vh, self.bh, self.wh]
#
#     def set_myself(self):
#         self.param([self, *self.vh])
#         self.param([self, *self.bh])
#         self.param([self, *self.wh])
#
#     def __call__(self, x, h0 = None):
#
#         '''
#         if "batch_first=True":
#             x shape: (B, T, E)
#             result shape: (B, T, N)
#         else:
#             x shape: (T, B, E)
#             result shape: (T, B, N)
#
#         self.input shape: (T, E, B)
#         self.output shape (T, N, B)
#         h0 shape: (H, B)
#         '''
#
#         batch_size, seq_len, embedding = x.shape()
#
#         if h0 == None:
#             h0 = Tensor.zeros(self.n_layers, batch_size, self.hidden_size)
#
#
#         x_n = x
#         h_t_1 = h0
#         h_t = h0
#         output = []
#         for t in range(seq_len):
#             for n in range(self.n_layers):
#                 h_t[n] = self.nonlinearity(x[t] @ self.vh[n] + h_t_1[n] @ self.wh[n] + self.bh[n])


        # self.mask = np.zeros((x.shape[1], self.N))
        # self.mask[-self.last_input:] = 1
        #
        # if self.batch_first:
        #     self.B = x.shape[0]
        #     self.T = x.shape[1]
        #     if h0 == None:
        #         self.h0 = np.zeros((self.H, self.B))
        #     self.input = np.transpose(x, axes=[1, 2, 0])
        #
        # else:
        #     self.B = x.shape[1]
        #     self.T = x.shape[0]
        #     if h0 == None:
        #         self.h0 = np.zeros((self.H, self.B))
        #     self.input = np.transpose(x, axes=[0, 2, 1])
        #
        # if self.nonlinearity == 'tanh':
        #     self.func = RNN.tanh_
        #     self.derivative = RNN.derivative_t
        # else:
        #     self.func = RNN.relu_
        #     self.derivative = RNN.derivative_r
        #
        # self.hidden_no_activation = np.zeros(((self.T, self.H, self.B)))
        # self.hidden_activation = np.zeros(((self.T, self.H, self.B)))
        #
        # if self.last_input == 0:
        #     self.last_input = self.T
        # self.output_no_activation = np.zeros(((self.last_input, self.N, self.B)))
        # self.output_activation = np.zeros(((self.last_input, self.N, self.B)))
        #
        # temp = self.h0 + 0
        # temp_ = 0
        # for t in range(self.T):
        #     temp_ = Parameter.calling[self][0] @ self.input[t] + Parameter.calling[self][1] @ temp + Parameter.calling[self][3].reshape(-1, 1)
        #     self.hidden_no_activation[t] = temp_ + 0
        #     temp = self.func(temp_)
        #     self.hidden_activation[t] = temp + 0
        #
        #     if (self.last_input == 0):
        #
        #         temp_ = Parameter.calling[self][2] @ temp + Parameter.calling[self][4].reshape(-1, 1)
        #         self.output_no_activation[t] = temp_ + 0
        #         temp__ = self.func(temp_)
        #         self.output_activation[t] = temp__ + 0
        #
        #     elif (t >= self.T - self.last_input):
        #         temp_ = Parameter.calling[self][2] @ temp + Parameter.calling[self][4].reshape(-1, 1)
        #         self.output_no_activation[t - (self.T - self.last_input)] = temp_ + 0
        #         temp__ = self.func(temp_)
        #         self.output_activation[t - (self.T - self.last_input)] = temp__ + 0
        #
        # if self.batch_first:
        #     return np.transpose(self.output_activation, axes=[2, 0, 1])
        # else:
        #     return np.transpose(self.output_activation, axes=[0, 2, 1])

class Flatten:
    def __init__(self):
        Parameter([self, []])

    def __call__(self, x):
        return x.reshape(x.shape[0], -1)

class Sigmoid:
    def __init__(self):
        Parameter([self,[]])
    def __call__(self, x):
        self.input = x + 0
        return 1 / (1 + Tensor.exp(-x))

class ReLU:
    def __init__(self):
        Parameter([self,[]])
    def __call__(self, x):
        return x * (1 + Tensor.sign(x)) / 2

class Leaky_ReLU:
    def __init__(self, a = 0.2):
        self.a = a
        Parameter([self, []])
    def __call__(self, x):
        return x * ((1 + Tensor.sign(x)) / 2 + self.a * (1 + Tensor.sign(-x)) / 2)

class Tanh:
    def __init__(self):
        Parameter([self,[]])

    def __call__(self, x):
        return (Tensor.exp(2 * x) - 1) / (Tensor.exp(2 * x) + 1)

class GeLU:
    def __init__(self):
        Parameter([self,[]])

    def __call__(self, x):
        return Tensor.gelu(x)
        #return 0.5 * x * (1 + Tensor.tanh(0.79788456 * (x + 0.044715 * (x ** 3))))

class ModuleList:

    def __init__(self, layers):
        self.layers = layers
        self.index = 0

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.layers):
            result = self.layers[self.index]
            self.index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, index):
        return self.layers[index]

class CrossEntropyLoss:
    def __init__(self, l1_reg = 0, l2_reg = 0, clip = None):
        self.backward_list = []
        self.predicted = None
        self.true = None
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def __call__(self, predicted, true):
        assert predicted.ndim == 2
        true = true.reshape(-1)
        if true.value.dtype != np.int_:
            true.value = np.array(true.value, dtype=np.int_)
        sub = predicted.max(axis=1, keepdims=True)
        sub.requires_grad = False
        self.logits = Tensor(predicted, local_gradients=predicted.local_gradients)
        self.predicted = Tensor.softmax(predicted - sub)
        if predicted.ndim == 1:
            self.number_of_classes = predicted.shape[0]
            self.true = Tensor((Tensor.arange(0, self.number_of_classes) == true) * 1, requires_grad=False)
            self.loss = -1 * Tensor.sum(self.true * Tensor.log(self.predicted + 1e-20), axis=0)
            return self
        else:
            self.number_of_classes = predicted.shape[-1]
            self.true = Tensor(np.eye(self.number_of_classes)[true.value], requires_grad=False)
            self.loss = Tensor(-1 * Tensor.sum(self.true * Tensor.log(self.predicted + 1e-20), axis=1))

            # Unusual loss like in pytorch for backprop calculation
            #self.false = Tensor(np.invert(self.true.value), requires_grad=False)
            #self.loss_ = -1 * self.true * Tensor.log(self.predicted  + 1e-10) + self.false * self.predicted
            return self

    def backward(self):
        self.analytics = (self.predicted - self.true) / self.predicted.shape[0] if self.predicted.ndim > 1 else self.predicted - self.true
        self.analytics = self.analytics.detach() * self.logits
        self.gradients = self.analytics.backward()

        #self.gradients = self.loss_.backward()

        global Parameter
        for index, layer in enumerate(Parameter.layers[::-1]):
            if type(layer).__name__ in ('Linear', 'Conv'):
                if self.gradients[layer.w].ndim == 3:
                    layer.w.grad += self.gradients[layer.w] / self.gradients[layer.w].shape[1]
                    layer.b.grad += self.gradients[layer.b] / self.gradients[layer.w].shape[1]
                else:
                    layer.w.grad += self.gradients[layer.w] / self.loss.shape[0]
                    layer.b.grad += self.gradients[layer.b] / self.loss.shape[0]
                layer.w.grad += self.l1_reg * Tensor.sign(layer.w).value + self.l2_reg * layer.w.value

            elif type(layer).__name__ in ('BatchNorm', 'LayerNorm'):
                layer.gamma.grad += self.gradients[layer.gamma]
                layer.beta.grad += self.gradients[layer.beta]

            elif type(layer).__name__ == 'Embedding':
                layer.w.grad += self.gradients[layer.w].mean(0)



