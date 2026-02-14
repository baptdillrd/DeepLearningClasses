import matplotlib.pyplot as plt

results = {
    "ResNet18": (11.17, 93.02),
    "DenseNet121": (6.95, 94.50),
    "VGG19": (139.57, 91.20),
    # Ajoute tes autres modèles ici...
}

plt.figure(figsize=(10, 7))

for model_name, (params, acc) in results.items():
    # Tracer le point
    plt.scatter(params, acc, s=100) 
    
    # Ajouter le texte à côté du point
    plt.text(params + 0.5, acc + 0.05, model_name, fontsize=9)

plt.title("Performance vs Complexité sur CIFAR10")
plt.xlabel("Nombre de paramètres (M)")
plt.ylabel("Top 1 Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.6)

# Ajuster les axes selon tes résultats
plt.xlim(0, max([p for p, a in results.values()]) * 1.2)
plt.ylim(min([a for p, a in results.values()]) - 1, 100)

plt.show()