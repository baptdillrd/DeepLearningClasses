import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import csv
import cifarutils as cu
from datetime import datetime
from cifarutils import trainloader, testloader
from models.utils import progress_bar
from models.resnet_light import *

# Définition du GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('==> Preparing models..')

teachernet = ResNet18()
teacher = teachernet.to(device)
teacher_path = './checkpoint/ckpt_ResNet-2026-03-22-Mixup-FP32.pth'
checkpoint_teacher = torch.load(teacher_path, map_location=device)
teacher.load_state_dict(checkpoint_teacher['net'])
teacher.eval()

net = ResNet18_Super_Light_DW_Quantizable()
student = net.to(device)
netname = f"Distill_Student_{net.__class__.__name__}"

temperature = 4.0
alpha = 0.7

criterion_cls = nn.CrossEntropyLoss()
criterion_kd = nn.KLDivLoss(reduction='batchmean')
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

n_epochs = 200
best_acc = 0
start_epoch = 0

def distillation_loss(student_outputs, teacher_outputs, labels, T, alpha, labels_b=None, lam=None):
    p_s = F.log_softmax(student_outputs / T, dim=1)
    p_t = F.softmax(teacher_outputs / T, dim=1)
    distill_loss = criterion_kd(p_s, p_t) * (T**2)
    
    if lam is not None and labels_b is not None:
        hard_loss = lam * criterion_cls(student_outputs, labels) + (1 - lam) * criterion_cls(student_outputs, labels_b)
    else:
        hard_loss = criterion_cls(student_outputs, labels)
    
    return alpha * distill_loss + (1.0 - alpha) * hard_loss

heure_fichier = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"../Experimentations/logs/distillation_logs_{heure_fichier}_{netname}.csv"

if not os.path.exists("../Experimentations/logs/"):
    os.makedirs("../Experimentations/logs/")

with open(log_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'test_loss', 'learning_rate', 
                     'train_acc', 'test_acc', 'training_time', 'testing_time', 'mixup_used'])

def train(epoch, use_mixup=True):
    print('\nEpoch: %d' % epoch)
    start_time = cu.gethour()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = cu.mixup_data(inputs, targets, alpha=1.0, device=device)
            mixup_used = "yes"
        else:
            targets_a, targets_b, lam = targets, None, None
            mixup_used = "no"

        outputs = net(inputs)
        
        with torch.no_grad():
            teacher_outputs = teacher(inputs)
        
        loss = distillation_loss(outputs, teacher_outputs, targets_a, temperature, alpha, targets_b, lam)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if use_mixup:
            correct += (lam * predicted.eq(targets_a).sum().item() + (1 - lam) * predicted.eq(targets_b).sum().item())
        else:
            correct += predicted.eq(targets_a).sum().item()

        avg_loss = train_loss / (batch_idx + 1)
        train_acc = 100.*correct/total
        progress_bar(batch_idx, len(trainloader), f'Loss: {avg_loss:.3f} | Acc: {train_acc:.2f}%')
        
    duration = cu.gethour() - start_time
    return avg_loss, train_acc, optimizer.param_groups[0]['lr'], duration, mixup_used

def test(epoch):
    global best_acc
    start_time = cu.gethour()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = net(inputs)
            
            loss = criterion_cls(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = test_loss / (batch_idx + 1)
            acc = 100.*correct/total
            
            progress_bar(batch_idx, len(testloader), f'Loss: {avg_loss:.3f} | Acc: {acc:.2f}%')

    if acc > best_acc:
        print(f'Saving best student.. Acc: {acc:.2f}%')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }
        if not os.path.isdir('checkpoint'): 
            os.mkdir('checkpoint')
        torch.save(state, f'./checkpoint/ckpt_{netname}_best.pth')
        best_acc = acc

    duration = cu.gethour() - start_time
    return acc, avg_loss, duration


heure_debut_globale = datetime.now()

for epoch in range(start_epoch, n_epochs):
    tr_loss, tr_acc, lr, tr_time, m_used = train(epoch)
    te_acc, te_loss, te_time = test(epoch)
    scheduler.step()
    
    with open(log_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, f"{tr_loss:.3f}", f"{te_loss:.3f}", lr, 
                         f"{tr_acc:.2f}", f"{te_acc:.2f}", tr_time, te_time, m_used])

print(f'Entraînement terminé. Temps total : {datetime.now() - heure_debut_globale}')