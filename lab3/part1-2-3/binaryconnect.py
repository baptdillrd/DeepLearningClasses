import torch.nn as nn
import numpy

class BC():
    def __init__(self, model):
        #Permet de compter le nombre de couche Conv2d et Linear
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
        self.saved_params = [] #Permet de sauvegarder les poids en full précision
        self.target_modules = [] #Va contenir la liste des poids à modifier
        self.model = model 

        #Permet de build la copie initiale de tous les paramètres et du module cible
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    #La référence est stockée au tenseur de poids
                    self.target_modules.append(m.weight)

    #Permet de sauvegarder les poids actuels
    def save_params(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):
        #On sauvegarde les paramètres full précision
        self.save_params()

        ### (2) Binarisation déterministe
        #On remplace les poids par leur signe (1 ou -1)
        for index in range(self.num_of_params):
            self.target_modules[index].data.sign_()
            self.target_modules[index].data[self.target_modules[index].data == 0] = 1

    #Permet de restaurer la copie de sauvegarde dans le modèle
    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
      
    #Permet de cliper les paramètres dans l'intervalle [-1,1]
    def clip(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.clamp_(-1, 1)

    #On appelle directement l'inférence via l'objet BC
    def forward(self, x):
        out = self.model(x)
        return out