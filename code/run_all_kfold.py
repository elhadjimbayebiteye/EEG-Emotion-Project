import os
import time

# Liste des configurations à tester
channel_list = [62, 48, 32, 16, 8, 6, 4]

for ch in channel_list:
    print(f"\n\n===== LANCEMENT DU K-FOLD POUR {ch} ÉLECTRODES =====\n\n")
    
    # Envoi de la valeur au main_kfold
    os.environ["TARGET_CHANNELS"] = str(ch)
    
    # On lance le script principal
    exit_code = os.system("python /home/elhadji/EEG_Emotion_Project/code/main_kfold.py")
    
    if exit_code != 0:
        print(f"ERREUR pendant l'exécution pour {ch} électrodes. On arrête.")
        break
    
    print(f"\n>>> FIN DU RUN {ch} ÉLECTRODES ! <<<\n")
    print("Pause de 10 secondes avant le suivant…")
    time.sleep(10)
