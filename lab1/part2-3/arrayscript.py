import torch
import matplotlib.pyplot as plt
from models import ResNet18, PreActResNet18, VGG, DenseNet121

# Modèles et noms
model_classes = [ResNet18, PreActResNet18, lambda: VGG("VGG16"), DenseNet121]
model_names = ["ResNet18", "PreActResNet18", "VGG16", "DenseNet121"]

# Calcul du nombre de paramètres
num_params = []

for model_cls in model_classes:
    m = model_cls()
    n = sum(p.numel() for p in m.parameters())
    num_params.append(n / 1e6)  # en millions
    print(f"Nombre de paramètres du modèle {model_cls} : {num_params} ")

# Accuracies
acc_th = [93.02, 95.11, 92.64, 95.04]
acc_exp = [, , , ]

## manual add
# acc.append(87.77)
# num_params.append(11.17)
# model_names.append("ResNet18 mixup lr=0.05")

# Plot figure théorique
plt.figure()
plt.scatter(num_params, acc_th)

for i, name in enumerate(model_names):
    plt.text(num_params[i], acc_th[i], name)

plt.xlabel("Nombre de paramètres (millions)")
plt.ylabel("Accuracy théorique (%)")
plt.title("Accuracy vs Nombre de paramètres")
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_vs_params_theorique.png", dpi=300)
print("Figure sauvegardée dans accuracy_vs_params_theorique.png")

plt.close()


# Plot figure expérimentale
plt.figure()
plt.scatter(num_params, acc_exp)

for i, name in enumerate(model_names):
    plt.text(num_params[i], acc_exp[i], name)

plt.xlabel("Nombre de paramètres (millions)")
plt.ylabel("Accuracy expérimentale (%)")
plt.title("Accuracy vs Nombre de paramètres")
plt.grid(True)

plt.tight_layout()
plt.savefig("accuracy_vs_params_experimental.png", dpi=300)
print("Figure sauvegardée dans accuracy_vs_params_experimental.png")

plt.close()