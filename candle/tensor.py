import numpy
from collections import defaultdict
from scipy.stats import norm
try:
    import cupy
except:
    pass


'''
Tensor class is a full analog of pytorch.tensor
New class overloads most popular numpy functions ( +, @, ones, reshape etc.)
So new class also provides all gradient computations with computational graph.
You can also run all computations on gpu (if available) by setting .to('gpu')
All gpu computations provided thanks to cupy library.
a big thanks to this repo: https://github.com/sradc/SmallPebble/blob/main/smallpebble/smallpebble.py#L708 , most of the cool ideas are taken from there
'''

np = numpy

class Tensor:

    def __init__(self, value, requires_grad=False, local_gradients=None, dtype=np.float64):

        if not isinstance(value, Tensor):
            self.value = np.array(value, dtype=dtype)
            if local_gradients:
                self.local_gradients = local_gradients
                self.requires_grad = True
            else:
                self.requires_grad = requires_grad
                self.local_gradients = None

        elif isinstance(value, Tensor):
            self.value = value.value
            if value.local_gradients:
                self.local_gradients = value.local_gradients
                self.requires_grad = True
            else:
                self.requires_grad = requires_grad
                self.local_gradients = None

        else:
            raise Exception("Incorrect input format")

        self.shape = self.value.shape
        self.ndim = self.value.ndim
        self.device = 'cpu'
        self.grad = 0

    def to(self, device = 'cpu'):
        global np
        if device == 'cpu':
            np = numpy
            if self.device == 'gpu':
                self.value = np.array(self.value.get())
            self.device = 'cpu'

        elif device == 'gpu':
            self.device = 'gpu'
            np = cupy
            self.value = np.array(self.value)
        else:
            raise Exception("No device has found")
        return self

    def to_cpu(self):
        self.device = 'cpu'
        self.value = numpy.array(self.value)
        return

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        value = self.value + other.value
        temp = []
        if self.requires_grad == True:
            if np.prod(np.array(self.shape)) >= np.prod(np.array(other.shape)):
                temp.append(('add', self, lambda x: x))
            else:
                def calc_loc_grad(x):
                    x = x * np.prod(np.array(other.shape)) / np.prod(np.array(self.shape))
                    while np.prod(np.array(x.shape)) != np.prod(np.array(self.shape)):
                        #x = x[0]
                        x = x.mean(0)
                    return x
                temp.append(('add', self, calc_loc_grad))

        if other.requires_grad == True:
            if np.prod(np.array(self.shape)) <= np.prod(np.array(other.shape)):
                temp.append(('add', other, lambda x: x))
            else:
                def calc_loc_grad(x):
                    x = x * np.prod(np.array(self.shape)) / np.prod(np.array(other.shape))
                    while np.prod(np.array(x.shape)) != np.prod(np.array(other.shape)):
                        #x = x[0]
                        x = x.mean(0)
                    return x
                temp.append(('add', other, calc_loc_grad))

        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __radd__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        value = self.value - other.value
        temp = []
        if self.requires_grad == True:
            if np.prod(np.array(self.shape)) >= np.prod(np.array(other.shape)):
                temp.append(('add', self, lambda x: x))
            else:
                def calc_loc_grad(x):
                    x = x * np.prod(np.array(other.shape)) / np.prod(np.array(self.shape))
                    while np.prod(np.array(x.shape)) != np.prod(np.array(self.shape)):
                        #x = x[0]
                        x = x.mean(0)
                    return x
                temp.append(('add', self, calc_loc_grad))

        if other.requires_grad == True:
            if np.prod(np.array(self.shape)) <= np.prod(np.array(other.shape)):
                temp.append(('add', other, lambda x: -x))
            else:
                def calc_loc_grad_(x):
                    x = x * np.prod(np.array(self.shape)) / np.prod(np.array(other.shape))
                    while np.prod(np.array(x.shape)) != np.prod(np.array(other.shape)):
                        #x = x[0]
                        x = x.mean(0)
                    return -x
                temp.append(('add', other, calc_loc_grad_))

        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __rsub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        value = self.value * other.value
        temp = []
        if self.requires_grad == True:
            if np.prod(np.array(self.shape)) >= np.prod(np.array(other.shape)):
                temp.append(('add', self, lambda x: x * other.value))
            else:
                def calc_loc_grad(x):
                    x = x * other.value
                    while np.prod(np.array(x.shape)) != np.prod(np.array(self.shape)):
                        x = x.sum(0)
                    return x
                temp.append(('add', self, calc_loc_grad))

        if other.requires_grad == True:
            if np.prod(np.array(self.shape)) <= np.prod(np.array(other.shape)):
                temp.append(('add', other, lambda x: x * self.value))
            else:
                def calc_loc_grad_(x):
                    x = x * self.value
                    while np.prod(np.array(x.shape)) != np.prod(np.array(other.shape)):
                        x = x.sum(0)
                    return x
                temp.append(('add', other, calc_loc_grad_))

        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __rmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other * self
        # value = self.value * other.value
        # temp = []
        # if self.requires_grad == True:
        #     temp.append(('rmul', self, lambda x: x * other.value))
        # if other.requires_grad == True:
        #     temp.append(('rmul', other, lambda x: x * self.value))
        # local_gradients = tuple(temp)
        # return Tensor(value, local_gradients=local_gradients)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        value = self.value @ other.value
        temp = []
        if self.requires_grad == True:
            temp.append(('matmul', self, lambda x: x @ other.value.swapaxes(-1, -2)))
        if other.requires_grad == True:
            temp.append(('matmul', other, lambda x: self.value.swapaxes(-1, -2) @ x))
        #local_gradients = (('matmul', self, lambda x: x @ other.value.T), ('matmul', other, lambda x: self.value.T @ x))
        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        value = other.value @ self.value
        #changed prove
        #local_gradients = (('rmatmul', other, lambda x: x @ self.value.T), ('rmatmul', self, lambda x: other.value.T @ x))
        temp = []
        if self.requires_grad == True:
            temp.append(('rmatmul', self, lambda x: other.value.T @ x))
        if other.requires_grad == True:
            temp.append(('rmatmul', other, lambda x: x @ self.value.T))
        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    @classmethod
    def inv(cls, a):
        if a.requires_grad == True:
            value = 1. / a.value
            local_gradients = (('inv', a, lambda x: x * -1. / (a.value ** 2)),)
        else:
            value = 1. / a.value
            #local_gradients = ((),)
            local_gradients = None

        return cls(value, local_gradients=local_gradients)

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor.__mul__(self, Tensor.inv(other))

    def __rtruediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return Tensor.__rmul__(Tensor.inv(self), other)

    def __neg__(self):
        return Tensor.__mul__(self, -1)

    def __pow__(self, n):
        value = self.value ** n
        local_gradients = (('pow', self, lambda x: x * np.ones(self.shape) * n * (self.value ** (n - 1))),)
        return Tensor(value, local_gradients=local_gradients)

    def __eq__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.value == other.value

    def __lt__(self, other):
        other = other if  isinstance(other, Tensor) else Tensor(other)
        return self.value < other.value

    def __le__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.value <= other.value

    def __gt__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.value > other.value

    def __ge__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.value >= other.value

    def __ne__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self.value != other.value

    def __index__(self):
        return int(self.value)

    def __getitem__(self, item):
        # What is happening here???
        if isinstance(item, int):
            temp = np.zeros(self.shape)
            temp[item] = 1
        else:
            temp = np.zeros(self.shape)
            temp[item] = 1

        def multiply_by_locgrad(path_value):
            _ = np.zeros(self.shape)
            _[item] = path_value
            return _

        local_gradients = (('getitem', self, multiply_by_locgrad),)
        return Tensor(self.value[item], local_gradients=local_gradients)

    def __setitem__(self, key, val):
        self.value[key] = val

    def detach(self):
        return Tensor(self.value, requires_grad=False)

    def transpose(self, *args):
        local_gradients = (('transpose', self, lambda x: x.transpose(*args)),)
        return Tensor(self.value.transpose(*args), local_gradients=local_gradients)

    @classmethod
    def sin(cls, a):
        value = np.sin(a.value)
        local_gradients = (
            ('sin', a, lambda x: x * np.cos(a.value)),
        )
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def cos(cls, a):
        value = np.cos(a.value)
        local_gradients = (
            ('cos', a, lambda x: x * -np.sin(a.value)),
        )
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def exp(cls, a):
        value = np.exp(a.value)
        local_gradients = (
            ('exp', a, lambda x: x * value),
        )
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def log(cls,a):
        value = np.log(a.value)
        local_gradients = (
            ('log', a, lambda x: x * 1. / a.value),
        )
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def zeros(cls, shape, requires_grad=False):
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @classmethod
    def clip(cls, input, min, max):
        value = np.clip(input.value, min, max)
        local_gradients = (
            ('clip', input, lambda x: x * (input.value >= min) * (input.value <= max)),
        )
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def triu(cls, input, diagonal=0):
        value = np.triu(input.value, k=diagonal)
        local_gradients = (
            ('triu', input, lambda x: x * np.triu(np.ones_like(input.value), k=diagonal)),
        )
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def tril(cls, input, diagonal=0):
        value = np.tril(input.value, k=diagonal)
        local_gradients = (
            ('triu', input, lambda x: x * np.tril(np.ones_like(input.value), k=diagonal)),
        )
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def sum(cls, array, axis=None, keepdims=False):
        if not keepdims:
            if axis is not None:
                local_gradients = (('sum', array, lambda x: np.expand_dims(np.array(x), axis=axis) + np.zeros(array.shape)),)
                return Tensor(np.sum(array.value, axis=axis), local_gradients=local_gradients)
            else:
                local_gradients = (('sum', array, lambda x: x + np.zeros(array.shape)),)
                return Tensor(np.sum(array.value, axis=axis), local_gradients=local_gradients)
        else:
            value = np.sum(array.value, axis=axis, keepdims=True) * np.ones_like(array.value)
            local_gradients = (('sum', array, lambda x: x),)
            return cls(value, local_gradients=local_gradients)


    def reshape(self, *args):
        local_gradients = (('reshape', self, lambda x: x.reshape(self.shape)),)
        return Tensor(self.value.reshape(*args), local_gradients=local_gradients)

    @classmethod
    def softmax(cls, z, axis=-1,):
        return Tensor.exp(z) / Tensor.sum(Tensor.exp(z), axis=axis, keepdims=True)

    @classmethod
    def arange(cls, stop, requires_grad=False):
        value = np.arange(stop)
        return cls(value, requires_grad=requires_grad)

    @classmethod
    def abs(cls, array):
        value = np.abs(array.value)
        local_gradients = (('abs', array, lambda x: x * np.sign(array.value)),)
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def sum_(cls, array, axis=None):
        value = np.sum(array.value, axis=axis, keepdims=True) * np.ones_like(array.value)
        local_gradients = (('sum_', array, lambda x: x),)
        return cls(value, local_gradients=local_gradients)

    @classmethod
    def sliding_window_view(cls, matrix, kernel_z, kernel_y, kernel_x):

        result = np.lib.stride_tricks.sliding_window_view(matrix.value, (1, kernel_z, kernel_y, kernel_x)).copy()

        def multiply_by_locgrad(path_value):
            temp = np.zeros(matrix.shape)
            if np.__name__ == 'numpy':
                np.add.at(np.lib.stride_tricks.sliding_window_view(temp, (1, kernel_z, kernel_y, kernel_x), writeable=True), None, path_value)
            elif np.__name__ == 'cupy':
                np.add.at(np.lib.stride_tricks.sliding_window_view(temp, (1, kernel_z, kernel_y, kernel_x)), None, path_value)

            return temp

        local_gradients = (('slide', matrix, multiply_by_locgrad),)
        return cls(result, local_gradients=local_gradients)

    @classmethod
    def ones(cls, shape, requires_grad=False):
        return cls(np.ones(shape), requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, x, requires_grad=False):
        return cls(np.ones_like(x.value), requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, x, requires_grad=False):
        return cls(np.zeros_like(x.value), requires_grad=requires_grad)

    @classmethod
    def randn(cls, shape, requires_grad=False, dtype=np.float64):
        return cls(np.random.normal(size=shape), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def uniform(cls, low=0.0, high=1.0, size=None, requires_grad=False, dtype=np.float64):
        return cls(np.random.uniform(low=low, high=high, size=size), requires_grad=requires_grad, dtype=dtype)

    @classmethod
    def int_(cls, *args):
        return cls(np.int_(*args))

    @classmethod
    def arange(cls, *args):
        return cls(np.arange(*args), dtype=int)

    @classmethod
    def sign(cls, a, requires_grad=False):
        value = np.sign(a.value)
        return cls(value, requires_grad=requires_grad)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return np.array_repr(self.value)

    @classmethod
    def sqrt(cls, a):
        return Tensor.__pow__(a, 1 / 2)

    def max(self, axis=0, keepdims=False):
        ## ADD LOCAL_GRADIENTS
        return Tensor(self.value.max(axis=axis, keepdims=keepdims))

    def min(self, axis=0, keepdims=False):
        ## ADD LOCAL_GRADIENTS
        return Tensor(self.value.min(axis=axis, keepdims=keepdims))

    @classmethod
    def topk(cls, array, k, axis=1):
        value1 = np.argsort(array.value)[:, -k:]
        value2 = np.take_along_axis(array.value, value1, axis=axis)
        return cls(value1, requires_grad=False), Tensor(value2, requires_grad=False)

    @classmethod
    def randint(cls, low, high, size, dtype=np.float64):
        value = np.random.randint(low, high, size)
        return cls(value, requires_grad=False, dtype=dtype)

    @classmethod
    def stack(cls, arrays, axis=0):
        value = np.stack([array.value for array in arrays], axis=axis)
        return cls(value, requires_grad=False)

    @classmethod
    def unsqueeze(cls, array, axis=0):
        value = np.expand_dims(array.value, axis=axis)
        return cls(value, requires_grad=False)

    @classmethod
    def repeat(cls, array, n, axis):
        value = np.repeat(array.value, n, axis)
        return cls(value, requires_grad=False)

    @classmethod
    def cat(cls, arrays, axis=0):
        if isinstance(arrays[0], Tensor):
            value = np.concatenate([array.value for array in arrays], axis=axis)
        else:
            value = np.concatenate([array for array in arrays], axis=axis)
        return cls(value, requires_grad=False)

    def tolist(self):
        return self.value.tolist()

    @classmethod
    def split(cls, array, split_size_or_sections, axis=0):
        value = np.split(array.value, split_size_or_sections, axis=axis)
        return cls(value, requires_grad=False)

    def multinomial_1d(array, num_samples):
        value = np.random.choice(np.arange(array.shape[-1]), size = num_samples, p = array.value.reshape(-1))
        return Tensor(value, requires_grad=False, dtype=int).reshape(1,-1)

    def multinomial(array, num_samples):
        return Tensor.cat([Tensor.multinomial_1d(one_array, num_samples) for one_array in array], axis=0)

    def multinomial_1d_from_array(indices, probs, num_samples):
        value = np.random.choice(indices.value, size = num_samples, p = probs.value.reshape(-1) / probs.value.sum())
        return Tensor(value, requires_grad=False, dtype=int).reshape(1,-1)

    def multinomial_from_array(indices, probs, num_samples):
        return Tensor.cat([Tensor.multinomial_1d_from_array(one_array1, one_array2, num_samples) for one_array1, one_array2 in zip(indices, probs)], axis=0)

    @classmethod
    def mean(cls, array, axis=None, keepdims=False):
        if axis == None:
            return Tensor.sum(array, axis=None, keepdims=keepdims) / np.size(array.value)
        else:
            delimeter = 1
            if not isinstance(axis, int):
                for ax in axis:
                    delimeter = delimeter * array.shape[ax]
            else:
                delimeter = array.shape[axis]

            return Tensor.sum(array, axis=axis, keepdims=keepdims) / delimeter

    @classmethod
    def std(cls, array, axis=None, keepdims=False):

        if axis == None or axis == 0:
            mean = Tensor.mean(array, axis=axis, keepdims=False)
            sub = array - mean
            squared = sub ** 2
            scaled_sum = Tensor.mean(squared, axis=axis, keepdims=keepdims)
            std = Tensor.sqrt(scaled_sum)
            return std

        elif axis >= 1:
            mean = Tensor.mean(array, axis=axis, keepdims=True)
            sub = array - mean
            squared = sub ** 2
            scaled_sum = Tensor.mean(squared, axis=axis, keepdims=keepdims)
            std = Tensor.sqrt(scaled_sum)
            out = std + 0
            out.local_gradients = (('std', std, lambda x: x * array.shape[axis] / (array.shape[axis] - 1)),) # array.shape[axis] / (array.shape[axis] - 1) additional multiplier due to dissimilarity
            return out

    def backward(self, value=None):
        value = Tensor.ones_like(self).value if value is None else value
        gradients = defaultdict(lambda: 0)
        def compute_gradients(variable, path_value):
            if variable.local_gradients:
                for oper_type, child, child_gradient_func in variable.local_gradients:
                    value_path_to_child = child_gradient_func(path_value)
                    gradients[child] += value_path_to_child
                    compute_gradients(child, value_path_to_child)

        compute_gradients(self, path_value=value)
        return gradients

    @classmethod
    def relu(cls, x):
        return x * (1 + Tensor.sign(x)) / 2

    @classmethod
    def leaky_relu(cls, x):
        return x * ((1 + Tensor.sign(x)) / 2 + 0.2 * (1 + Tensor.sign(-x)) / 2)

    @classmethod
    def tanh(cls, x):
        return (Tensor.exp(2 * x) - 1) / (Tensor.exp(2 * x) + 1)

    @classmethod
    def gelu(cls, x):
        try:
            return x * norm.cdf(x.value)
        except:
            return x * norm.cdf(x.value.get())
        # too long
        #return 0.5 * x * (1 + Tensor.tanh(0.79788456 * (x + 0.044715 * (x ** 3))))

    @classmethod
    def sigmoid(cls, x):
        return 1 / (1 + Tensor.exp(-x))