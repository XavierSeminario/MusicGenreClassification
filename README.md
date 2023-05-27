[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11101312&assignment_repo_type=AssignmentRepo)
# XNAP-Project title (replace it by the title of your project)
You must have installed 7zip on your computer
Objective: Classify music in 163 genres, either based on pre-extracted features, or by analysing tracks from mp3 samples
Type of Model: CNN
Task: Audio Classification
Data: Metadata and extracted features (342MB), 8,000 tracks in 8 genres (7 GB)
Write here a short summary about your project. The text must include a short introduction and the targeted goals

## Code structure
1. models.py : es defineixen varies arquitectures de xarxes neuronals usant 'torch.nn'. Aquesta consta de sis classes.
2. utils.py : és una col·lecció de funcions i classes que realitzen diverses tasques relacionades amb el processament de dades per un model de classificació de gèneres de música usant espectogrames d'audio.
3. download_data.sh : crea un directori anomenat "data" si no existeix, descarrega i extreu dos fitxers zip ("fma_metadata.zip" i "fma_small.zip") d'una URL proporcionada i realitza altres operacions per organitzar els fitxers descarregats dins del directori "data".
4. environment.yml : arxiu de configuració en YAML que descriu un entorn conda amb el nom "xnap-example".
5. main.py : realitza el procés d'entrenament i prova d'un model de xarxa neuronal convolucional per a la classificació de gèneres musicals utilitzant dades d'espectrogrames. També utilitza la plataforma WandB per al seguiment i la visualització.
6. train.py : defineix un procés d'entrenament per a un model de xarxa neuronal, calcula les pèrdues per a cada lot d'entrenament i les registra en el registre de WandB per al seguiment i la visualització.

You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Example Code
The given code is a simple CNN example training on the MNIST dataset. It shows how to set up the [Weights & Biases](https://wandb.ai/site)  package to monitor how your network is learning, or not.

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/XNAP-Project/environment.yml) file has all the required dependencies. Run the following command: ``conda env create --file environment.yml `` to create a conda environment with all the required dependencies and then activate it:
```
conda activate xnap-example
```

To run the example code:
```
chmod +x Descarregar_Dades.sh
./Descarregar_Dades.sh

python main.py
```



## Contributors
-
- Manuel Arnau Fernández -> 1597487@uab.cat
- Pau Fuentes Hernández -> pau.fuentesh@autonoma.cat
- Andrea González Aguilera -> andrea.gonzaleza@autonoma.cat
- Xavier Seminario Monllaó -> 1603853@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Computational Mathematics & Data analyitics, 
UAB, 2023
