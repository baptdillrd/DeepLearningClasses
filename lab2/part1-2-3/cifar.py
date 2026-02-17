import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import os
from matplotlib import pyplot as plt
import numpy as np

import csv
from datetime import datetime
from models.utils import progress_bar
from models.resnet import *
from models.densenet import *
from models.preact_resnet import *
from models.vgg import *

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



#Définition du modèle
print('==> Building model..')
net = ResNet18()
# net = DenseNet121()
# net = PreActResNet18()
# net = VGG('VGG16')
net = net.to(device)
netname = net.__class__.__name__

#Définition des paramètres
criterion = torch.nn.CrossEntropyLoss() #définition de la fonction de perte
#définition de l'optimiseur (modifie les poids du réseau en fonction de la loss)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
n_epochs = 5  #nombre d'époques

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
start_epoch = 0 #époque de départ
best_acc = 0 #Définition de la meilleure accuracy


#Fonction de resume
resume = False      #A changer en True si on charge un modèle existant

if resume:
    print('==> Resuming from checkpoint..')
    ckpt_path = f'./checkpoint/ckpt_{net.__class__.__name__}.pth'
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded {ckpt_path} at epoch {start_epoch} (Best Acc: {best_acc:.2f}%)")
    else:
        print("Error: no checkpoint directory found!")


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


#Création du fichier de logs
heure_fichier = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"../Experimentations/logs/logs_{heure_fichier}_{netname}.csv"

#Header du logfile
with open(log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'test_loss', 'learning_rate', 'train_acc', 'test_acc', 'training_time', 'testing_time', 'mixup_used']) # En-têtes

def gethour():
    return datetime.now()
heurepretraining = gethour()


#Figure pour chaque batch
f = plt.figure(figsize=(10,10))

for i,(data,target) in enumerate(trainloader):
    
    data = (data.numpy())
    print(data.shape)
    plt.subplot(2,2,1)
    plt.imshow(data[0].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,2)
    plt.imshow(data[1].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,3)
    plt.imshow(data[2].swapaxes(0,2).swapaxes(0,1))
    plt.subplot(2,2,4)
    plt.imshow(data[3].swapaxes(0,2).swapaxes(0,1))

    break

f.savefig(f'../Experimentations/batchplot/DA_{heure_fichier}.png')



print("Début de l'entraînement...")

def train(epoch, use_mixup=True):
    print('\nEpoch: %d' % epoch)
    start_time = gethour()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_batches = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if use_mixup:
            #On appelle la fonction Mixup
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0, device=device)
            outputs = net(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())
            mixup_used = "yes"
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            mixup_used = "no"

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        avg_loss = train_loss / (batch_idx + 1)
        train_acc = 100.*correct/total

        msg = f'Loss: {avg_loss:.3f} | Acc: {train_acc:.2f}%'
        progress_bar(batch_idx, total_batches, msg)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (avg_loss, train_acc, correct, total))
        
    duration = gethour() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    return avg_loss, train_acc, current_lr, duration, mixup_used


def test(epoch):
    global best_acc
    start_time = gethour()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_acc = 0
    total_batches = len(trainloader)

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

            avg_loss = test_loss / (batch_idx + 1)

            msg = f'Loss: {test_loss/(batch_idx+1):.3f} | Acc: {test_acc:.2f}%'
            progress_bar(batch_idx, total_batches, msg)
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)\n'
            #              % (test_loss/(batch_idx+1), test_acc , correct, total))
        # Save checkpoint.
        if test_acc > best_acc:
            print(f'Saving best model - Accuracy : {test_acc:.2f} %')
            state = {
                'net': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'acc': test_acc,
                'epoch': epoch,
                'net_name' : net.__class__.__name__
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'./checkpoint/ckpt_{net.__class__.__name__}.pth')
            best_acc = test_acc
        
        duration = gethour() - start_time 
        return test_acc, avg_loss, duration




#Boucle principale
for epoch in range(start_epoch, n_epochs):
    tr_loss, tr_acc, lr_rate, hrtrainepoch, mixup_used= train(epoch)
    te_acc, te_loss, hrtestepoch = test(epoch)
    scheduler.step()

    with open(log_path, mode = 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{tr_loss:.3f}", f"{te_loss:.3f}", lr_rate, f"{tr_acc:.2f}", f"{te_acc:.2f}", hrtrainepoch, hrtestepoch, mixup_used])


print('Entraînement terminé.')



#Définition du temps d'entrainement, du nom du modèle utilisé et du nombre de paramètres
heureposttraining = datetime.now()
trainingtime = heureposttraining - heurepretraining
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

#Affichage de tout ça
print(f"Nombre de paramètres : {num_params}")
print(f"Nom du modèle utilisé : {netname}")
print(f"Temps d'entrainement : {trainingtime}")
