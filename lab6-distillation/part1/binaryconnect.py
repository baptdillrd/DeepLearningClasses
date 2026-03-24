import torch.nn as nn
import numpy

class BC():
    def __init__(self, model):
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1
        start_range = 0
        end_range = count_targets - 1
        self.bin_range = numpy.linspace(start_range,
                                     end_range, end_range - start_range + 1)\
                                     .astype('int').tolist()

        self.num_of_params = len(self.bin_range)
        self.saved_params = [] 
        self.target_modules = []
        self.model = model 

        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):
        self.save_params()

        for index in range(self.num_of_params):
            self.target_modules[index].data.sign_()
            self.target_modules[index].data[self.target_modules[index].data == 0] = 1

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
      
    def clip(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp_(-1, 1)

    def forward(self, x):
        out = self.model(x)
        return out