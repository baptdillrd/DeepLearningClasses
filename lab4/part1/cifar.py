import torch
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn.utils.prune as prune
import os
import numpy as np
import csv

from cifarutils import trainloader, testloader
from matplotlib import pyplot as plt
from binaryconnect import BC
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



#Définition du modèle
print('==> Building model..')
net = ResNet18()
# net = DenseNet121()
# net = PreActResNet18()
# net = VGG('VGG16')
net = net.to(device)
netname = net.__class__.__name__
net_BC = BC(net)


#Définition des paramètres
criterion = torch.nn.CrossEntropyLoss() #définition de la fonction de perte
#définition de l'optimiseur (modifie les poids du réseau en fonction de la loss)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
n_epochs = 100  #nombre d'époques

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
start_epoch = 0 #époque de départ
best_acc = 0 #Définition de la meilleure accuracy

#Fonction de resume
resume = False      #A changer en True si on charge un modèle existant
if resume:
    print('==> Resuming from checkpoint..')
    ckpt_path = f'./weight_used/.pth'       #A compléter si on utilise resume
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
    writer.writerow(['epoch', 'train_loss', 'test_loss', 'learning_rate', 'train_acc', 'test_acc', 'training_time', 'testing_time', 'mixup_used', 'binary_connect_used']) # En-têtes

#Définition de l'heure du début d'entrainement
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
def train(epoch, use_mixup=True, use_BC = False):
    print('\nEpoch: %d' % epoch)
    start_time = gethour()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    total_batches = len(trainloader)
    bc_used = "yes" if use_BC else "no"

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_BC:
            net_BC.binarization()

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
        if use_BC:
            net_BC.restore()
        optimizer.step()

        if use_BC:
            net_BC.clip()

        train_loss += loss.item()

        avg_loss = train_loss / (batch_idx + 1)
        train_acc = 100.*correct/total

        msg = f'Loss: {avg_loss:.3f} | Acc: {train_acc:.2f}%'
        progress_bar(batch_idx, total_batches, msg)
        
    duration = gethour() - start_time
    current_lr = optimizer.param_groups[0]['lr']
    return avg_loss, train_acc, current_lr, duration, mixup_used, bc_used

def test(epoch, use_mixup=True, use_BC = False):
    global best_acc
    start_time = gethour()
    timestamp = datetime.now().strftime("%H%M%S")
    if use_BC:
        net_BC.binarization()
    binary = "BC" if use_BC else "FP32"
    mixup = "Mixup" if use_mixup else "NoMixup"
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
        # Save checkpoint.
        if test_acc > best_acc:
            if use_BC : net_BC.restore()
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
            torch.save(state, f'./checkpoint/ckpt_{net.__class__.__name__}-{timestamp}-{mixup}-{binary}.pth')
            best_acc = test_acc
            if use_BC : net_BC.binarization()

        duration = gethour() - start_time 
        return test_acc, avg_loss, duration

#Boucle principale
for epoch in range(start_epoch, n_epochs):
    tr_loss, tr_acc, lr_rate, hrtrainepoch, mixup_used, bc_used= train(epoch)
    te_acc, te_loss, hrtestepoch = test(epoch)
    scheduler.step()
    with open(log_path, mode = 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{tr_loss:.3f}", f"{te_loss:.3f}", lr_rate, f"{tr_acc:.2f}", f"{te_acc:.2f}", hrtrainepoch, hrtestepoch, mixup_used, bc_used])
print('Entraînement terminé.')

#Définition du temps d'entrainement, du nom du modèle utilisé et du nombre de paramètres
heureposttraining = datetime.now()
trainingtime = heureposttraining - heurepretraining
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

#Affichage de tout ça
print(f"Nombre de paramètres : {num_params}")
print(f"Nom du modèle utilisé : {netname}")
print(f"Temps d'entrainement : {trainingtime}")
