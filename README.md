[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11101312&assignment_repo_type=AssignmentRepo)
# Classificació de gèneres musicals Grup 11

L'objectiu d'aquest projecte és classificar música en 163 gèneres diferents utilitzant atributs extrets o analitzant les pistes dels arxius MP3. Per aconseguir-ho, es farà servir una combinació de xarxes neuronals convolucionals (CNN) i xarxes neuronals recurrents (RNN).

Els models de CNN són excel·lents per processar dades espaials com imatges i, en aquest cas, es poden utilitzar per extreure característiques de les pistes de música. Les capes de convolució de la CNN poden aprendre patrons locals en les dades d'àudio i, posteriorment, s'aplicaran capes totalment connectades per classificar les característiques extretes.
D'altra banda, les RNN són útils per processar seqüències de dades, com ara les representacions en temps de les pistes d'àudio. Les RNN són capaces de capturar dependències a llarg termini en les dades seqüencials, cosa que les fa adequades per a tasques de classificació d'àudio.

El projecte en el qual s'ha treballat té com a tasca principal la classificació d'àudio.

Pel que fa a les dades, es disposa de metadades que ocupen aproximadament 342MB i inclouen informació com el títol de la cançó, l'artista, l'àlbum, etc. A més, hi ha 8.000 cançons de 8 gèneres diferents que ocupen al voltant de 7GB. Aquestes cançons seran fonamentals per entrenar i avaluar el model de classificació.

<p align="center">
<img src="https://github.com/DCC-UAB/xnap-project-matcad_grup_11/blob/main/IMG_20230307_230625460.jpg", widht="300", height="300">
</p>

## Estructura del codi
1. models.py : Es defineixen varies arquitectures de xarxes neuronals usant 'torch.nn'. Aquesta consta de tres classes.
2. utils.py : És una col·lecció de funcions i classes que realitzen diverses tasques relacionades amb el processament de dades per un model de classificació de gèneres de música usant espectogrames d'àudio.
3. download_data.sh : Crea un directori anomenat "data" si no existeix, descarrega i extreu dos fitxers zip ("fma_metadata.zip" i "fma_small.zip") d'una URL proporcionada i realitza altres operacions per organitzar els fitxers descarregats dins del directori "data".
4. environment.yml : Arxiu de configuració en YAML que descriu un entorn conda amb el nom "xnap-example".
5. main.py : Realitza el procés d'entrenament i prova d'un model de xarxa neuronal convolucional per a la classificació de gèneres musicals utilitzant dades d'espectrogrames. També utilitza la plataforma WandB per al seguiment i la visualització. Si no estan ja creats, també utilitza funcions de utils.py per crear els espectrogrames.
6. train.py : Defineix un procés d'entrenament per a un model de xarxa neuronal, calcula les pèrdues per a cada lot d'entrenament i les registra en el registre de WandB per al seguiment i la visualització.
7. test.py : Defineix el procés de testeig per a un model de xarxa neuronal que es realitzarà a cada època, igual que en el fitxer train, es calculen les mètriques pertinents i es registren al WandB.

Abans d'executar el codi cal crear un entorn local amb conda i activar-lo. El fitxer [environment.yml] conté totes les dependències necessàries per a poder executar el codi adequadament. Per activar-lo cal executar ``conda env create --file environment.yml``. També cal tenir instal·lat 7zip a l'ordinador ``sudo apt install p7zip-full p7zip-rar``.

```
conda activate xnap-example

chmod +x download_data.sh
```
Per executar un model:
```
python main.py
```

## Contribuïdors
- Manuel Arnau Fernández -> 1597487@uab.cat
- Pau Fuentes Hernández -> pau.fuentesh@autonoma.cat
- Andrea González Aguilera -> andrea.gonzaleza@autonoma.cat
- Xavier Seminario Monllaó -> 1603853@uab.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Matemàtica Computacional i Analítica de Dades, 
UAB, 2023
