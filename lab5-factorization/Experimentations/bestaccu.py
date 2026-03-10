import pandas as pd
import os
import glob

#On regarde les logs dans le fichier de logs
log_folder = '../Experimentations/logs/*.csv'
list_of_files = glob.glob(log_folder)

if not list_of_files:
    print("Aucun fichier de logs trouvé.")
    exit()

#On trie les fichiers du plus récent au plus ancien
list_of_files.sort(key=os.path.getctime, reverse=True)

#On garde les 10 derniers fichiers de logs
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


def get_best_metrics(file_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return

    #On charge les donnés du CSV
    df = pd.read_csv(file_path)

    #Calcul de la meilleure accuracy de test
    best_test_idx = df['test_acc'].idxmax()
    best_test_acc = df.loc[best_test_idx, 'test_acc']
    epoch_test = df.loc[best_test_idx, 'epoch']

    #Calcul de la meilleure accuracy d'entrainement
    best_train_idx = df['train_acc'].idxmax()
    best_train_acc = df.loc[best_train_idx, 'train_acc']
    epoch_train = df.loc[best_train_idx, 'epoch']

    #Calcul du temps total d'entrainement pour le modèle
    train_times = pd.to_timedelta(df['training_time'])
    test_times = pd.to_timedelta(df['testing_time'])
    total_time = train_times.sum() + test_times.sum()

    #Affichage des accuracy et du temps d'entrainement en fonction du nombre d'epoch
    print(f"--- Résultats pour : {os.path.basename(file_path)} ---")
    print(f"Meilleure Accuracy TEST  : {best_test_acc:.2f}% (Époque {int(epoch_test)})")
    print(f"Meilleure Accuracy TRAIN : {best_train_acc:.2f}% (Époque {int(epoch_train)})")
    print(f"Temps total cumulé       : {total_time}")
    print("-" * 40)

get_best_metrics(selected_file)