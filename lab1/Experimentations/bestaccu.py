import pandas as pd
import os
import glob
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


def get_best_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return

    # Chargement du CSV
    df = pd.read_csv(file_path)

    # 1. Meilleure Accuracy de TEST (ce que tu présenteras en priorité)
    best_test_idx = df['test_acc'].idxmax()
    best_test_acc = df.loc[best_test_idx, 'test_acc']
    epoch_test = df.loc[best_test_idx, 'epoch']

    # 2. Meilleure Accuracy de TRAIN
    best_train_idx = df['train_acc'].idxmax()
    best_train_acc = df.loc[best_train_idx, 'train_acc']
    epoch_train = df.loc[best_train_idx, 'epoch']

    # 3. Calcul du temps total d'entraînement (somme des colonnes de temps)
    # On convertit les chaînes '0:03:39' en objets timedelta pour sommer
    train_times = pd.to_timedelta(df['training_time'])
    test_times = pd.to_timedelta(df['testing_time'])
    total_time = train_times.sum() + test_times.sum()

    print(f"--- Résultats pour : {os.path.basename(file_path)} ---")
    print(f"Meilleure Accuracy TEST  : {best_test_acc:.2f}% (Époque {int(epoch_test)})")
    print(f"Meilleure Accuracy TRAIN : {best_train_acc:.2f}% (Époque {int(epoch_train)})")
    print(f"Temps total cumulé       : {total_time}")
    print("-" * 40)

# Utilisation : remplace par le nom de ton fichier réel
get_best_metrics(selected_file)