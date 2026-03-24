import torch
from models.resnet_light import *
from models.mobilenetv2 import *
from cifarutils import testloader

device = 'cpu'

net = ResNet18_Super_Light_DW_Quantizable()

path = "./checkpoint/ckpt_3b_ResNetSuperLight_0.9_pruned_final.pth"
print(path)

checkpoint = torch.load(
    path,
    map_location='cpu',
    weights_only=False
)

state_dict = checkpoint["net"]

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    new_state_dict[k] = v

net.load_state_dict(new_state_dict)

print("Epoch :", checkpoint["epoch"])
print("Best acc sauvegardée :", checkpoint["acc"])


def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100 * correct / total


n_params = sum(p.numel() for p in net.parameters())
print("Nombre total de paramètres :", n_params)

net.eval()

torch.backends.quantized.engine = 'qnnpack' 
net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(net, inplace=True)

print("\nCalibration...")
with torch.no_grad():
    for i, (inputs, _) in enumerate(testloader):
        net(inputs)
        if i >= 10:
            break

torch.quantization.convert(net, inplace=True)
print("Modèle quantizé ✓")

acc_int8 = evaluate_accuracy(net, testloader)
print(f"Accuracy INT8  : {acc_int8:.2f}%")
#print(f"Perte de précision : {acc_fp32 - acc_int8:.2f}%")

torch.save(
    {'net': net.state_dict(), 'acc': checkpoint['acc'], 'epoch': checkpoint['epoch']},
    './checkpoint/ckpt_18_custom_super_quant.pth'
)