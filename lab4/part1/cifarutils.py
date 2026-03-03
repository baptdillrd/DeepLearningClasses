import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn.utils.prune as prune
import os
import numpy as np
import csv

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