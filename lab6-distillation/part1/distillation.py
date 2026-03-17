import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cifarutils as cu
from models.resnet_fac import ResNet18Factorized
from models.mobilenetv2 import MobileNetV2
from models.utils import progress_bar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Chargement du TEACHER (Ton modèle déjà entraîné)
teacher = ResNet18Factorized().to(device)
checkpoint = torch.load('./checkpoint/ckpt_ResNet-2026-03-11-Mixup-FP32.pth', map_location=device)
teacher.load_state_dict(checkpoint['net'])
teacher.eval() # Important : Toujours en eval

# Création de l'ÉTUDIANT (Plus léger)
student = MobileNetV2().to(device)

# T > 1 lisse les probabilités pour révéler les relations entre classes [cite: 108, 120]
temperature = 4.0 
# Poids entre la CrossEntropy classique et la Distillation [cite: 103]
alpha = 0.7 

optimizer = optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion_cls = nn.CrossEntropyLoss()
# KLDivLoss pour comparer les "Soft Labels" [cite: 106, 118]
criterion_kd = nn.KLDivLoss(reduction='batchmean') 

def distillation_loss(student_outputs, teacher_outputs, labels, T, alpha):
    # 1. Perte classique (Hard labels)
    soft_loss = criterion_cls(student_outputs, labels)
    
    # 2. Perte de distillation (Soft labels) [cite: 103, 107]
    # On applique la température sur les logits avant le softmax
    p_s = F.log_softmax(student_outputs / T, dim=1)
    p_t = F.softmax(teacher_outputs / T, dim=1)
    
    # On multiplie par T^2 pour compenser la réduction de l'amplitude des gradients [cite: 109, 121]
    distill_loss = criterion_kd(p_s, p_t) * (T**2)
    
    return alpha * distill_loss + (1.0 - alpha) * soft_loss


def train_student(epoch):
    student.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(cu.trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward de l'étudiant
        student_outputs = student(inputs)
        
        # Forward du maître (sans calcul de gradient pour gagner de la mémoire)
        with torch.no_grad():
            teacher_outputs = teacher(inputs)
        
        # Calcul de la perte hybride
        loss = distillation_loss(student_outputs, teacher_outputs, targets, temperature, alpha)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = student_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(cu.trainloader), f'Loss: {train_loss/(batch_idx+1):.3f} | Acc: {100.*correct/total:.2f}%')

# Lancement
for epoch in range(20):
    train_student(epoch)