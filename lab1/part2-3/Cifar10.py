from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import csv

#Création du fichier de logs
heure_fichier = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = "../Experimentations"
log_file = "logs_{heure_fichier}.csv"


list_losses = []
train_accuracies = []
test_accuracies = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Normalization adapted for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Transforms is a list of transformations applied on the 'raw' dataset before the data is fed to the network.
# Here, Data augmentation (RandomCrop and Horizontal Flip) are applied to each batch, differently at each epoch, on the training set data only
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

### The data from CIFAR10 are already downloaded in the following folder
rootdir = '/opt/img/effdl-cifar10/'

c10train = CIFAR10(rootdir,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir,train=False,download=True,transform=transform_test)

trainloader = DataLoader(c10train,batch_size=32,shuffle=True)
testloader = DataLoader(c10test,batch_size=32)


## number of target samples for the final dataset
num_train_examples = len(c10train)
num_samples_subset = 15000

## We set a seed manually so as to reproduce the results easily
seed  = 2147483647

## Generate a list of shuffled indices ; with the fixed seed, the permutation will always be the same, for reproducibility
indices = list(range(num_train_examples))
np.random.RandomState(seed=seed).shuffle(indices)## modifies the list in place

## We define the Subset using the generated indices
c10train_subset = torch.utils.data.Subset(c10train,indices[:num_samples_subset])
print(f"Initial CIFAR10 dataset has {len(c10train)} samples")
print(f"Subset of CIFAR10 dataset has {len(c10train_subset)} samples")

# Finally we can define anoter dataloader for the training data
trainloader_subset = DataLoader(c10train_subset,batch_size=32,shuffle=True)
### You can now use either trainloader (full CIFAR10) or trainloader_subset (subset of CIFAR10) to train your networks.

from densenet import DenseNet121
net = DenseNet121().to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

n_epochs = 5

heurepretraining = datetime.now()


print("Début de l'entraînement...")

for epoch in range(n_epochs):

    # ======== TRAIN ========
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for data in trainloader_subset:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Accuracy train
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(trainloader_subset)
    train_accuracy = 100 * correct_train / total_train

    list_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy)

    # ======== VALIDATION / TEST ========
    net.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    test_accuracies.append(test_accuracy)

    # ======== AFFICHAGE ========
    print(
        f"Epoch {epoch+1}/{n_epochs} | "
        f"Train Loss: {epoch_loss:.3f} | "
        f"Train Acc: {train_accuracy:.2f}% | "
        f"Test Acc: {test_accuracy:.2f}%"
    )

    with open(log_path, mode = 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{epoch_loss:.3f}", f"{train_accuracy:.2f}", f"{test_accuracy:.2f}"])

print('Entraînement terminé.')

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"Nombre de paramètres : {num_params}")

heureposttraining = datetime.now()
trainingtime = heureposttraining - heurepretraining

print(f"Temps d'entrainement : {trainingtime}")

#print("train", train_accuracies)
#print("valid", test_accuracies)


epochs = range(1, len(list_losses) + 1)

plt.figure()
plt.plot(epochs, list_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("loss_plot.png")
plt.show()



epochs = range(1, len(train_accuracies) + 1)

plt.figure()
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, test_accuracies, label="Test Accuracy")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig("accuracy_plot.png")
plt.show()








