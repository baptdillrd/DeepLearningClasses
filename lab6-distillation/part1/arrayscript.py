import matplotlib.pyplot as plt

data = {
    "ResNet18 Vanilla": {"acc": 96.01, "score" : 3.98},
    "ResNet18 90% Pruned": {"acc": 93.2, "score": 0.521},
    "MobileNetV2 Vanilla": {"acc": 91.96, "score": 0.736},
    "MobileNetV2 80% Pruned DW": {"acc": 90.58, "score": 0.102},
    "ResNet18 W/2 80% Pruned DW": {"acc": 91.6, "score": 0.077},
    "ResNet W/4 50% Pruned DW": {"acc": 88.4, "score": 0.0073},
    "ResNet W/8 DW": {"acc": 75.3, "score": 0.0028}
}

names = list(data.keys())
scores = [d["score"] for d in data.values()]
accuracies = [d["acc"] for d in data.values()]

plt.figure(figsize=(10, 7))
plt.scatter(scores, accuracies, color='blue', s=80, alpha=0.7)
plt.axhline(y=90, color='red', linestyle='--', linewidth=2, label="Seuil 90%")
plt.text(min(scores), 90.5, "90% accu", color='red', fontweight='bold')
for i, name in enumerate(names):
    plt.text(scores[i], accuracies[i] + 0.5, name, fontsize=9, ha='center')

plt.xscale('log')
plt.xlabel("Score (Log Scale)")
plt.ylabel("Accuracy (%)")
plt.title("Compromis Accuracy vs Score des modèles")
plt.grid(True, which="both", ls="-", alpha=0.5)

plt.tight_layout()
plt.savefig("accuracy_vs_score.png", dpi=300)
print("Graphique sauvegardé sous 'accuracy_vs_score.png'")
plt.show()