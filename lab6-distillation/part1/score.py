import torch
import math
from torchinfo import summary
from models.resnet_fac import *
from models.mobilenetv2 import *
from models.resnet_light import *
from cifarutils import testloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#model = ResNet18().to(device)
model = Loicnet().to(device)

path = "./checkpoint/ckpt_3b_LoicNet0.7_pruned_final.pth"
print(path)
checkpoint = torch.load(
    path,
    map_location=device,
    weights_only=False
)


state_dict = checkpoint["net"]

new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        k = k[7:]
    new_state_dict[k] = v

model.load_state_dict(new_state_dict)

is_halfed = True

if is_halfed:
    model.half()

def evaluate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device).to(dtype)
            targets = targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return 100 * correct / total

acc = evaluate_accuracy(model, testloader)
print(f"Accuracy sur test set : {acc:.2f}%")

dtypes= [next(model.parameters()).dtype]
dummy = torch.randn(1,3,32,32).to(device)
dummy = dummy.to(dtypes[0])

stats = summary(
    model,
    input_data=dummy,
    dtypes=dtypes,
    verbose=0
)

ps = 0 # default 0
pu = 0 # default 0 (c'est le pruning)
qw = 8 # default 32 bits (quantization à la fin)
qa = 8 # default 32 bits
w = stats.total_params
print(f"total parameters : {w}")

param_div = 5.6e06
f =  stats.total_mult_adds
ops_div = 2.8e08

score = ((1 - (ps + pu))*(qw / 32)*w) / param_div + ((1 - ps)*((max(qw, qa)) / 32)*f) / ops_div

print(f"{path} model has score : {score}")