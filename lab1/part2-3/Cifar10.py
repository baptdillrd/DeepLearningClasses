import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import os

import csv
from datetime import datetime
from models.utils import progress_bar
from models.resnet import *
from models.densenet import *
from models.preact_resnet import *
from models.vgg import *

#Création du fichier de logs
heure_fichier = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"../Experimentations/logs/logs_{heure_fichier}.csv"

#Header du logfile
with open(log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'learning_rate', 'train_acc', 'test_acc', 'training_time', 'testing_time']) # En-têtes

#Instanciation des listes de données 
list_losses = []
train_accuracies = []
test_accuracies = []

#Définition du GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Centrage des données sur la base du modèle
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

#Modification des données d'entrainement et normalisation des données d'entrainement et de test
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
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

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)

#Définition du modèle
print('==> Building model..')
# net = ResNet18()
# net = DenseNet121()
net = PreActResNet18()
# net = VGG('VGG16')
net = net.to(device)

#Définition des paramètres
criterion = torch.nn.CrossEntropyLoss() #définition de la fonction de perte
#définition de l'optimiseur (modifie les poids du réseau en fonction de la loss)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
n_epochs = 30  #nombre d'époques

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
start_epoch = 0 #époque de départ
best_acc = 0 #Définition de la meilleure accuracy

def gethour():
    return datetime.now()

heurepretraining = gethour()


print("Début de l'entraînement...")

def train(epoch):
    print('\nEpoch: %d' % epoch)
    start_time = gethour()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        avg_loss = train_loss / (batch_idx + 1)
        train_acc = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (avg_loss, train_acc, correct, total))
        
    duration = gethour() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    return avg_loss, train_acc, current_lr, duration


def test(epoch):
    global best_acc
    start_time = gethour()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_acc = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), test_acc , correct, total))
        
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        
        duration = gethour() - start_time 
        return test_acc, duration




#Boucle principale
for epoch in range(start_epoch, n_epochs):
    tr_loss, tr_acc, lr_rate, hrtrainepoch = train(epoch)
    te_acc, hrtestepoch = test(epoch)
    scheduler.step()

    with open(log_path, mode = 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{tr_loss:.3f}", lr_rate, f"{tr_acc:.2f}", f"{te_acc:.2f}", hrtrainepoch, hrtestepoch])


print('Entraînement terminé.')



#Définition du temps d'entrainement, du nom du modèle utilisé et du nombre de paramètres
heureposttraining = datetime.now()
trainingtime = heureposttraining - heurepretraining
netname = net.__class__.__name__
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

#Affichage de tout ça
print(f"Nombre de paramètres : {num_params}")
print(f"Nom du modèle utilisé : {netname}")
print(f"Temps d'entrainement : {trainingtime}")
