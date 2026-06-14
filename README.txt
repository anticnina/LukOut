LukOut - prepoznavanje jestivih i nejestivih biljaka iz familije luka
Autorke: Nina Antic (IN25/2023), Tamara Sevo (IN27/2023)

POTREBNO OKRUZENJE
-------------------
- Python 3.9 - 3.12 (TensorFlow trenutno NE radi na Python 3.13/3.14)

POTREBNE BIBLIOTEKE
--------------------
Instalacija (u terminalu, u folderu projekta):

    pip install tensorflow numpy matplotlib scikit-learn pillow tqdm requests

Koriscene biblioteke po skriptama:
- tensorflow / keras   -> izgradnja i trening neuronskih mreza (lukout.py, lukout_nasaNN.py)
- numpy                -> rad sa nizovima/podacima
- matplotlib           -> grafici (istorija treninga, matrice konfuzije, ROC krive)
- scikit-learn         -> metrike (accuracy, F1, precision, recall, ROC/AUC)
- pillow (PIL)         -> ucitavanje i obrada slika (skripte/)
- tqdm                 -> progres bar pri preuzimanju/pripremi slika (skripte/)
- requests             -> preuzimanje slika sa GBIF-a (skripte/plant_image_collector.py)

STRUKTURA PROJEKTA
-------------------
LukOut/
  Dataset/                    -> pripremljen skup slika (train/val/test, edible/inedible)
  lukout_nasaNN.py             -> CNN treniran od nule (nasa mreza)
  lukout.py                   -> glavni model (EfficientNetB0, transfer learning)
  lukout_izvestaj.py           -> zajednicke funkcije za evaluaciju i grafike
  skripte/
    plant_image_collector.py  -> preuzimanje slika sa GBIF-a
    dataset_prepare.py        -> priprema/augmentacija i podela na train/val/test

POKRETANJE (redosled)
----------------------
Dataset/ folder NIJE ukljucen u repozitorijum (preveliki je), zato
ga je potrebno prvo generisati pokretanjem skripti iz foldera skripte/.

1. Preuzimanje slika sa GBIF-a:
       python skripte/plant_image_collector.py
   - Pravi folder "dataset/" sa podfolderima edible/ i inedible/.

2. Priprema skupa podataka (skaliranje, augmentacija, podela train/val/test):
       python skripte/dataset_prepare.py
   - Cita iz "dataset/", pravi folder "dataset_prepared/"
     (train/val/test, svaki sa edible/ i inedible/).
   - Preimenovati "dataset_prepared" u "Dataset" (lukout.py i
     lukout_nasaNN.py ocekuju tacno taj naziv foldera).

3. Treniranje i evaluacija modela "od nule" (nasa mreza):
       python lukout_nasaNN.py

4. Treniranje i evaluacija glavnog modela (EfficientNetB0):
       python lukout.py

Skripti iz koraka 3 i 4 automatski cuvaju grafike i izvestaje u
folderima "results_nasaNN/" i "results/" (podfolderi prag_0.5/ i prag_0.9/).
