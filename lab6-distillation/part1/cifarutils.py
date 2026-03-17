import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn.utils.prune as prune
import os
import numpy as np
import csv
from datetime import datetime
from matplotlib import pyplot as plt

#Centrage des données sur la base du modèle
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#Modification des données d'entrainement et normalisation des données d'entrainement et de test
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    normalize_scratch,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])


#Importation des données de CIFAR10 et définition des loader de test et de train
rootdir = '/opt/img/effdl-cifar10/'
c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)
trainloader = DataLoader(c10train,batch_size=64,shuffle=True)
testloader = DataLoader(c10test,batch_size=64)


#Fonction de mixup (à ajouter ou non durant l'entrainement)
#Le mixup retourne les entrées mixées, les deux paires de cibles et lambda
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

#Calcule la perte mixée
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


#Définition de l'heure du début d'entrainement
def gethour():
    return datetime.now()
heurepretraining = gethour()




# #Figure pour chaque batch
# f = plt.figure(figsize=(10,10))

# for i,(data,target) in enumerate(trainloader):
    
#     data = (data.numpy())
#     print(data.shape)
#     plt.subplot(2,2,1)
#     plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(2,2,2)
#     plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(2,2,3)
#     plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
#     plt.subplot(2,2,4)
#     plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))

#     break
# f.savefig(f'../Experimentations/batchplot/DA_{heure_fichier}.png')