import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune

from models import ResNet18, ResNet18_Light
from cifar10_load import trainloader, testloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_accuracy(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100 * correct / total


###################################
# 1. Charger modèle
###################################

net = ResNet18().to(device)

checkpoint = torch.load(
    "./checkpoint/ckpt_3.pth",
    map_location=device
)

state_dict = checkpoint["net"]

# Enlever le "module." si présent
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]   # enlever "module."
    new_state_dict[k] = v

net.load_state_dict(new_state_dict)

print("Checkpoint chargé")
print("Acc :", checkpoint["acc"])



###################################
# 2. Appliquer L1 pruning
###################################

prune_amount = 0.99
modules_to_prune = []

for module in net.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(
            module,
            name="weight",
            amount=prune_amount
        )
        modules_to_prune.append(module)

print("Pruning appliqué (masque actif)")

acc_masked = test_accuracy(net)

print("Accuracy masked :", acc_masked)


###################################
# 4. Fine tuning AVEC masque
###################################

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    net.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)


def train_epoch():
    net.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


best_acc = 0
n_epochs = 5
for epoch in range(1, n_epochs):
    train_epoch()
    acc = test_accuracy(net)
    scheduler.step()
    print("Epoch", epoch, "Acc:", acc)
    if acc > best_acc:
        best_acc = acc
        torch.save({
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }, "./checkpoint/ckpt_3b_pruned_masked_finetuned.pth")

        print("Checkpoint masked finetuned sauvegardé")


###################################
# 5. Remove pruning (final)
###################################

for module in modules_to_prune:
    prune.remove(module, 'weight')

print("Pruning rendu permanent")


###################################
# 6. Sauvegarde finale propre
###################################

torch.save({
    'net': net.state_dict(),
    'acc': best_acc,
    'epoch': n_epochs
}, "./checkpoint/ckpt_3b_pruned_final.pth")

print("Modèle final sauvegardé")