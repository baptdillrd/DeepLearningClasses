import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from models.resnet import ResNet18 #importation du modèle entrainé

#Configuration du GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Préparation des données
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

rootdir = '/opt/img/effdl-cifar10/'
c10test = torchvision.datasets.CIFAR10(rootdir, train=False, download=True, transform=transform_test)
testloader = DataLoader(c10test, batch_size=100, shuffle=False)

#Reconstitution de l'architecture du modèle
print('==> Loading architecture..')
net = ResNet18()
net = net.to(device)
net.half()

#Chargement des points sauvegardés
ckpt_path = './checkpoint/ckpt_ResNet.pth' #Nom du fichier à adapter en fonction du modèle
print(f'==> Loading weights from {ckpt_path}..')

checkpoint = torch.load(ckpt_path, map_location=device)
net.load_state_dict(checkpoint['net'])

#Mode évaluation
net.eval() 

#Test de précision
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device).half(), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

final_acc = 100.*correct/total
print(f'Précision du modèle chargé sur les données de test : {final_acc:.2f}%')
print(f'Précision enregistrée dans le checkpoint était : {checkpoint["acc"]:.2f}%')