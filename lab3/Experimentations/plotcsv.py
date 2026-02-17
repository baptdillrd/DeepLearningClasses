import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil

#On regarde les logs dans le fichier de logs
log_folder = '../Experimentations/logs/*.csv'
list_of_files = glob.glob(log_folder)

if not list_of_files:
    print("Aucun fichier de logs trouvé.")
    exit()

#On trie les fichiers du plus récent au plus ancien
list_of_files.sort(key=os.path.getctime, reverse=True)

#On garde les 10 derniers fichiers
recent_files = list_of_files[:10]

print("\n--- Choisissez le fichier de logs à tracer ---")
for i, file in enumerate(recent_files):
    #On affiche le nom du fichier et son index pour mieux comprendre
    print(f"[{i}] {os.path.basename(file)}")

#boucle try pour le renvoi de la valeur de l'index du fichier par l'utilisateur
try:
    choice = int(input(f"\nEntrez le numéro (0-{len(recent_files)-1}) [Défaut 0] : ") or 0)
    selected_file = recent_files[choice]
except (ValueError, IndexError):
    print("Choix invalide, utilisation du fichier le plus récent.")
    selected_file = recent_files[0]

print(f"\nTraitement de : {selected_file}")

#On crée le dossier de plot
filename = os.path.basename(selected_file)
timestamp = filename.replace("logs_", "").replace(".csv", "")
plot_dir = f"./plots/{timestamp}"
if os.path.exists(plot_dir):
    shutil.rmtree(plot_dir)  #On écrase le dossier s'il existe déjà
os.makedirs(plot_dir)

#On charge les données du fichier de logs
df = pd.read_csv(selected_file)

#Graphique du Learning Rate
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['learning_rate'], color='tab:blue', linewidth=2)
plt.title(f"Évolution du Learning Rate ({timestamp})")
plt.xlabel("Époque")
plt.ylabel("Learning Rate")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{plot_dir}/learning_rate.png")
plt.close()

#Graphique de la Loss (train vs test)
plt.figure(figsize=(10, 5))
plt.plot(df['epoch'], df['train_loss'], color='tab:red', marker='o', markersize=4)
plt.plot(df['epoch'], df['test_loss'], color='tab:blue', marker='o', markersize=4)
plt.title("Évolution de la Perte (Train Loss vs Test Loss)")
plt.xlabel("Époque")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(f"{plot_dir}/train_loss.png")
plt.close()

#Graphique de l'accuracy (train vs test)
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