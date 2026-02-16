import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil

# --- 1. Sélection interactive du fichier de logs ---
log_folder = '../Experimentations/logs/*.csv'
list_of_files = glob.glob(log_folder)

if not list_of_files:
    print("Aucun fichier de logs trouvé.")
    exit()

# Trier les fichiers par date de création (du plus récent au plus ancien)
list_of_files.sort(key=os.path.getctime, reverse=True)

# Garder les 10 derniers
recent_files = list_of_files[:10]

print("\n--- Choisissez le fichier de logs à tracer ---")
for i, file in enumerate(recent_files):
    # On affiche l'index et le nom du fichier pour plus de clarté
    print(f"[{i}] {os.path.basename(file)}")

try:
    choice = int(input(f"\nEntrez le numéro (0-{len(recent_files)-1}) [Défaut 0] : ") or 0)
    selected_file = recent_files[choice]
except (ValueError, IndexError):
    print("Choix invalide, utilisation du fichier le plus récent.")
    selected_file = recent_files[0]

print(f"\nTraitement de : {selected_file}")

# 2. Extraction de la date du nom du fichier pour créer le dossier
filename = os.path.basename(selected_file)
timestamp = filename.replace("logs_", "").replace(".csv", "")

# Définition et création du dossier de destination
plot_dir = f"./plots/{timestamp}"
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)  # On écrase si déjà existant
os.makedirs(plot_dir)

# 3. Chargement des données
df = pd.read_csv(selected_file)

# --- PLOT 1 : Learning Rate ---
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['learning_rate'], color='tab:blue', linewidth=2)
plt.title(f"Évolution du Learning Rate ({timestamp})")
plt.xlabel("Époque")
plt.ylabel("Learning Rate")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{plot_dir}/learning_rate.png")
plt.close() # Ferme la figure pour libérer la mémoire

# --- PLOT 2 : Train Loss ---
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_loss'], color='tab:red', marker='o', markersize=4)
plt.title("Évolution de la Perte (Train Loss)")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{plot_dir}/train_loss.png")
plt.close()

# --- PLOT 3 : Accuracies ---
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_acc'], label="Train Acc", color='forestgreen', marker='s', markersize=4)
plt.plot(df['epoch'], df['test_acc'], label="Test Acc", color='orange', marker='s', markersize=4)
plt.title("Comparaison des Précisions (Train vs Test)")
plt.xlabel("Époque")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{plot_dir}/accuracies.png")
plt.close()

print(f"Terminé ! Les 3 graphiques sont enregistrés dans : {plot_dir}")