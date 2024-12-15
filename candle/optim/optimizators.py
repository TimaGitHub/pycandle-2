import numpy

try:
    import cupy
except:
    pass

'''
note you can use here mm.Parameter for the latest Parameter object
and self.model._constructor_Parameter for the certain model
by default i use here model._constructor_Parameter
'''

np = numpy


class SGD:
    def __init__(self, model, lr=2e-4):
        self.model = model
        self.lr = lr
        global np
        if model.device == 'cpu':
            np = numpy
        else:
            np = cupy

    def step(self):
        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ in ('Linear', 'Conv', 'RNN', 'Embedding'):
                if layer.w.grad.ndim == 3:
                     layer.w.grad = layer.w.grad.mean(0)
                layer.w.value -= self.lr * layer.w.grad

            elif type(layer).__name__ in ('BatchNorm', 'LayerNorm'):
                layer.gamma.value -= self.lr * layer.gamma.grad
                layer.beta.value -= self.lr * layer.beta.grad

            if type(layer).__name__ == 'Linear':
                # check tne need of broadcating
                layer.b.value -= self.lr * layer.b.grad.mean(axis=0) if layer.b.requires_grad else 0
                #layer.b.value -= self.lr * layer.b.grad if layer.b.requires_grad else 0

            elif type(layer).__name__ == 'Conv':
                #check tne need of broadcating
                layer.b.value -= self.lr * layer.b.grad.mean(axis=(0, 2, 3), keepdims=True).squeeze(0) if layer.b.requires_grad else 0

            # elif type(layer).__name__ == 'RNN':
            #     self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] - self.lr * loss.backward_list[index][0],
            #                                 self.model._constructor_Parameter.calling[layer][1] - self.lr * loss.backward_list[index][1],
            #                                 self.model._constructor_Parameter.calling[layer][2] - self.lr * loss.backward_list[index][2],
            #                                 self.model._constructor_Parameter.calling[layer][3] - self.lr * loss.backward_list[index][3],
            #                                 self.model._constructor_Parameter.calling[layer][4] - self.lr * loss.backward_list[index][4]
            #                                 ]

class NAG:
    def __init__(self, model, lr=2e-4, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        global np
        if model.device == 'cpu':
            np = numpy
        else:
            np = cupy
        self.last_grad_w = None
        for layer in self.model._constructor_Parameter.layers:
            if type(layer).__name__ in  ('Linear', 'Conv', 'RNN', 'BatchNorm', 'LayerNorm', 'Embedding'):
                for sub_layer in layer.all_layers:
                    sub_layer.last_grad = 0

    def step(self):

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):

            if type(layer).__name__ in  ('Linear', 'Conv', 'RNN', 'Embedding'):
                if layer.w.grad.ndim == 3:
                    layer.w.grad = layer.w.grad.mean(0)

                layer.w.last_grad = - self.lr * layer.w.grad + self.momentum * layer.w.last_grad
                layer.w.value += layer.w.last_grad

            elif type(layer).__name__ in ('BatchNorm', 'LayerNorm'):
                layer.gamma.last_grad = - self.lr * layer.gamma.grad + self.momentum * layer.gamma.last_grad
                layer.gamma.value += layer.gamma.last_grad
                layer.beta.last_grad = - self.lr * layer.beta.grad + self.momentum * layer.beta.last_grad
                layer.beta.value += layer.beta.last_grad

            if type(layer).__name__ == 'Linear':
                layer.b.last_grad = - self.lr * layer.b.grad.mean(axis=0) + self.momentum * layer.b.last_grad if layer.b.requires_grad else 0
                layer.b.value += layer.b.last_grad

            elif type(layer).__name__ == 'Conv':
                layer.b.last_grad = - self.lr * layer.b.grad.mean(axis=(0, 2, 3), keepdims=True).squeeze(0) + self.momentum * layer.b.last_grad if layer.b.requires_grad else 0
                layer.b.value += layer.b.last_grad

            # elif type(layer).__name__ == 'RNN':
            #     self.last_grad_U[index] = - self.lr * loss.backward_list[index][0] + self.momentum * self.last_grad_U[index]
            #     self.last_grad_W[index] = - self.lr * loss.backward_list[index][1] + self.momentum * self.last_grad_W[index]
            #     self.last_grad_V[index] = - self.lr * loss.backward_list[index][2] + self.momentum * self.last_grad_V[index]
            #     self.last_grad_bh[index] = - self.lr * loss.backward_list[index][3] + self.momentum * self.last_grad_bh[index]
            #     self.last_grad_by[index] = - self.lr * loss.backward_list[index][4] + self.momentum * self.last_grad_by[index]
            #
            #     self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] + self.last_grad_U[index],
            #                                 self.model._constructor_Parameter.calling[layer][1] + self.last_grad_U[index],
            #                                 self.model._constructor_Parameter.calling[layer][2] + self.last_grad_U[index],
            #                                 self.model._constructor_Parameter.calling[layer][3] + self.last_grad_U[index],
            #                                 self.model._constructor_Parameter.calling[layer][4] + self.last_grad_U[index]
            #                                 ]

class RMSProp:
    def __init__(self, model, lr=2e-4, ro=0.99):
        global np
        if model.device == 'cpu':
            np = numpy
        else:
            np = cupy
        self.model = model
        self.lr = lr
        if ro < 0 or ro > 1:
            raise Exception("Incorrect ro value")
        self.ro = ro
        for layer in self.model._constructor_Parameter.layers:
            if type(layer).__name__ in ('Linear', 'Conv', 'RNN', 'BatchNorm', 'LayerNorm', 'Embedding'):
                for sub_layer in layer.all_layers:
                    sub_layer.grad_velocity = 0

    def step(self):

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ in  ('Linear', 'Conv', 'Embedding'):
                if layer.w.grad.ndim == 3:
                    layer.w.grad = layer.w.grad.mean(0)
                layer.w.grad_velocity = self.ro * layer.w.grad_velocity + (1 - self.ro) * layer.w.grad ** 2
                layer.w.value -= self.lr * layer.w.grad / np.sqrt(layer.w.grad_velocity + 1e-10)

            elif type(layer).__name__ in ('BatchNorm', 'LayerNorm'):
                layer.gamma.grad_velocity = self.ro * layer.gamma.grad_velocity + (1 - self.ro) * layer.gamma.grad ** 2
                layer.gamma.value -= self.lr * layer.gamma.grad / np.sqrt(layer.gamma.grad_velocity + 1e-10)

                layer.beta.grad_velocity = self.ro * layer.beta.grad_velocity + (1 - self.ro) * layer.beta.grad ** 2
                layer.beta.value -= self.lr * layer.beta.grad / np.sqrt(layer.beta.grad_velocity + 1e-10)

            if type(layer).__name__ == 'Linear':
                layer.b.grad_velocity = self.ro * layer.b.grad_velocity + (1 - self.ro) * layer.b.grad.mean(axis=0) ** 2  if layer.b.requires_grad else 0
                if layer.b.requires_grad:
                    layer.b.value -= self.lr * layer.b.grad.mean(axis=0) / np.sqrt(layer.b.grad_velocity + 1e-10)

            elif type(layer).__name__ == 'Conv':
                layer.b.grad_velocity = self.ro * layer.b.grad_velocity + (1 - self.ro) * layer.b.grad.mean(axis=(0, 2, 3), keepdims=True).squeeze(0) ** 2 if layer.b.requires_grad else 0
                if layer.b.requires_grad:
                    layer.b.value -= self.lr * layer.b.grad.mean(axis=(0, 2, 3), keepdims=True).squeeze(0) / np.sqrt(layer.b.grad_velocity + 1e-10)


            # elif type(layer).__name__ == 'RNN':
            #     self.grad_velocity_U[index] =  self.ro * self.grad_velocity_U[index] + (1 - self.ro) * loss.backward_list[index][0] ** 2
            #     self.grad_velocity_W[index] =  self.ro * self.grad_velocity_W[index] + (1 - self.ro) * loss.backward_list[index][1] ** 2
            #     self.grad_velocity_V[index] =  self.ro * self.grad_velocity_V[index] + (1 - self.ro) * loss.backward_list[index][2] ** 2
            #     self.grad_velocity_bh[index] =  self.ro * self.grad_velocity_bh[index] + (1 - self.ro) * loss.backward_list[index][3] ** 2
            #     self.grad_velocity_by[index] =  self.ro * self.grad_velocity_by[index] + (1 - self.ro) * loss.backward_list[index][4] ** 2
            #
            #     self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] - self.lr * loss.backward_list[index][0] / np.sqrt(self.grad_velocity_U[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][1] - self.lr * loss.backward_list[index][1] / np.sqrt(self.grad_velocity_W[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][2] - self.lr * loss.backward_list[index][2] / np.sqrt(self.grad_velocity_V[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][3] - self.lr * loss.backward_list[index][3] / np.sqrt(self.grad_velocity_bh[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][4] - self.lr * loss.backward_list[index][4] / np.sqrt(self.grad_velocity_wby[index] + 1e-5),
            #                                 ]

class ADAM:
    def __init__(self, model, lr=2e-4, momentum=0.9, ro=0.999):
        self.model = model
        global np
        if model.device == 'cpu':
            np = numpy
        else:
            np = cupy
        self.lr = lr
        self.momentum = momentum
        if ro < 0 or ro > 1:
            raise Exception("Incorrect ro value")
        self.ro = ro

        for layer in self.model._constructor_Parameter.layers:
            if type(layer).__name__ in ('Linear', 'Conv', 'RNN', 'BatchNorm', 'LayerNorm', 'Embedding'):
                for sub_layer in layer.all_layers:
                    sub_layer.grad_velocity = 0
                    sub_layer.last_grad = 0

    def step(self):

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):

            if type(layer).__name__ in ('Linear', 'Conv', 'Embedding'):
                if layer.w.grad.ndim == 3:
                    layer.w.grad = layer.w.grad.mean(0)
                layer.w.last_grad = - self.lr * layer.w.grad + self.momentum * layer.w.last_grad
                layer.w.grad_velocity = self.ro * layer.w.grad_velocity + (1 - self.ro) * layer.w.grad ** 2
                layer.w.value += layer.w.last_grad / np.sqrt(layer.w.grad_velocity + 1e-10)

            elif type(layer).__name__ in ('BatchNorm', 'LayerNorm'):
                layer.gamma.last_grad = - self.lr * layer.gamma.grad + self.momentum * layer.gamma.last_grad
                layer.gamma.grad_velocity = self.ro * layer.gamma.grad_velocity + (1 - self.ro) * layer.gamma.grad ** 2
                layer.gamma.value += layer.gamma.last_grad / np.sqrt(layer.gamma.grad_velocity + 1e-10)

                layer.beta.last_grad = - self.lr * layer.beta.grad + self.momentum * layer.beta.last_grad
                layer.beta.grad_velocity = self.ro * layer.beta.grad_velocity + (1 - self.ro) * layer.beta.grad ** 2
                layer.beta.value += layer.beta.last_grad / np.sqrt(layer.beta.grad_velocity + 1e-10)

            if type(layer).__name__ == 'Linear':
                layer.b.last_grad = - self.lr * layer.b.grad.mean(axis=0) + self.momentum * layer.b.last_grad if layer.b.requires_grad else 0
                layer.b.grad_velocity = self.ro * layer.b.grad_velocity + (1 - self.ro) * layer.b.grad.mean(axis=0) ** 2 if layer.b.requires_grad else 0
                if layer.b.requires_grad:
                    layer.b.value += layer.b.last_grad / np.sqrt(layer.b.grad_velocity + 1e-10)

            elif type(layer).__name__ == 'Conv':
                layer.b.last_grad = - self.lr * layer.b.grad.mean(axis=(0, 2, 3), keepdims=True).squeeze(0) + self.momentum * layer.b.last_grad if layer.b.requires_grad else 0
                layer.b.grad_velocity = self.ro * layer.b.grad_velocity + (1 - self.ro) * layer.b.grad.mean(axis=(0, 2, 3), keepdims=True).squeeze(0) ** 2 if layer.b.requires_grad else 0

                if layer.b.requires_grad:
                    layer.b.value += layer.b.last_grad / np.sqrt(layer.b.grad_velocity + 1e-10)

            # elif type(layer).__name__ == 'RNN':
            #     self.grad_velocity_U[index] = self.ro * self.grad_velocity_U[index] + (1 - self.ro) * loss.backward_list[index][0] ** 2
            #     self.grad_velocity_W[index] = self.ro * self.grad_velocity_W[index] + (1 - self.ro) * loss.backward_list[index][1] ** 2
            #     self.grad_velocity_V[index] = self.ro * self.grad_velocity_V[index] + (1 - self.ro) * loss.backward_list[index][2] ** 2
            #     self.grad_velocity_bh[index] = self.ro * self.grad_velocity_bh[index] + (1 - self.ro) * loss.backward_list[index][3] ** 2
            #     self.grad_velocity_by[index] = self.ro * self.grad_velocity_by[index] + (1 - self.ro) * loss.backward_list[index][4] ** 2
            #     self.last_grad_U[index] = - self.lr * loss.backward_list[index][0] + self.momentum * self.last_grad_U[index]
            #     self.last_grad_W[index] = - self.lr * loss.backward_list[index][1] + self.momentum * self.last_grad_W[index]
            #     self.last_grad_V[index] = - self.lr * loss.backward_list[index][2] + self.momentum * self.last_grad_V[index]
            #     self.last_grad_bh[index] = - self.lr * loss.backward_list[index][3] + self.momentum * self.last_grad_bh[index]
            #     self.last_grad_by[index] = - self.lr * loss.backward_list[index][4] + self.momentum * self.last_grad_by[index]
            #
            #     self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] + self.last_grad_U[index] / np.sqrt(self.grad_velocity_U[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][1] + self.last_grad_W[index] / np.sqrt(self.grad_velocity_W[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][2] + self.last_grad_V[index] / np.sqrt(self.grad_velocity_V[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][3] + self.last_grad_bh[index] / np.sqrt(self.grad_velocity_bh[index] + 1e-5),
            #                                 self.model._constructor_Parameter.calling[layer][4] + self.last_grad_by[index] / np.sqrt(self.grad_velocity_by[index] + 1e-5),
            #                                 ]
