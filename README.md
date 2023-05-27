[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11101312&assignment_repo_type=AssignmentRepo)
# Classificació de gèneres musicals

Objectiu: Classificar música en 163 gèneres, basat en atributs extrets o analitzant tracks d'arxius mp3.
Tipus de model: CNN i RNN.
Tasca: Classificació d'audio.
Dades: Metadata i atributs (342MB), 8,000 tracks de les cançons (7 GB).

## Code structure
1. models.py : Es defineixen varies arquitectures de xarxes neuronals usant 'torch.nn'. Aquesta consta de sis classes.
2. utils.py : és una col·lecció de funcions i classes que realitzen diverses tasques relacionades amb el processament de dades per un model de classificació de gèneres de música usant espectogrames d'audio.
3. download_data.sh : Crea un directori anomenat "data" si no existeix, descarrega i extreu dos fitxers zip ("fma_metadata.zip" i "fma_small.zip") d'una URL proporcionada i realitza altres operacions per organitzar els fitxers descarregats dins del directori "data".
4. environment.yml : Arxiu de configuració en YAML que descriu un entorn conda amb el nom "xnap-example".
5. main.py : Realitza el procés d'entrenament i un model de xarxa neuronal per a la classificació de gèneres musicals utilitzant dades d'espectrogrames. També utilitza la plataforma WandB per al seguiment i la visualització del procés.
6. train.py : Defineix un procés d'entrenament per a un model de xarxa neuronal, calcula les pèrdues per a cada lot d'entrenament i les registra en el registre de WandB per al seguiment i la visualització.
7. test.py : Defineix el procés de testeig per a un model de xarxa neuronal que es realitzarà a cada època, igual que en el fitxer train, es calculen les mètriques pertinents i es registren al WandB.

Abans d'executar el codi cal crear un entorn local amb conda i activar-lo. El fitxer [environment.yml] conté totes les dependències necessàries per a poder executar el codi adequadament. Per activar-lo cal executar ``conda env create --file environment.yml ``. També cal tenir instal·lat 7zip a l'ordinador ``sudo apt install p7zip-full p7zip-rar``.

```
conda activate xnap-example
```

Per executar un model:
```
chmod +x download_data.sh

python main.py
```

## Contribuïdors
-
- Manuel Arnau Fernández -> 1597487@uab.cat
- Pau Fuentes Hernández -> pau.fuentesh@autonoma.cat
- Andrea González Aguilera -> andrea.gonzaleza@autonoma.cat
- Xavier Seminario Monllaó -> 1603853@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Matemàtica Computacional i Analítica de Dades, 
UAB, 2023
