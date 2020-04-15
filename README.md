# Image Featurization

This is the featurization step for the project "Inferring Music And Visual Art Style Evolution via Computational Intelligence" (1).

Here we extracted the features from the penultimate layer of the network implemented by the authors Adrian Lecoutre, Benjamin Negrevergne and Florian Yger (2)(3), without performing any retraining procedures.

1. https://github.com/erikagardini/InferringMusicAndVisualArtStyleEvolution/edit/master/README.md

2. Adrian Lecoutre, Benjamin Negrevergne and Florian Yger (2017). Recognizing Art Style Automatically in painting with deep learning. _JMLR: Workshop and Conference Proceedings_ __(80)__ 1–17.

3. https://github.com/bnegreve/rasta

## Install python requirements

You can install python requirements with

```
pip3 install -r requirements.txt
```

## Downloads files

```
git clone https://github.com/erikagardini/ImageFeaturization.git
```

## Downloads the Wikipainting dataset

Download the full wikipaintings dataset (the one from WikiArt) executing the following commands. Warning: the file is about ~20GiB.

```
cd datasets
wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/wikipaintings_full.tgz
tar xzvf wikipaintings_full.tgz
cd ../
```

## Downloads the rasta models

```
cd models
wget www.lamsade.dauphine.fr/~bnegrevergne/webpage/software/rasta/rasta_models.tgz
tar xzvf rasta_models.tgz
cd ../
```

## How to use the code

### Step 1: Test the model

Test the model obtained during the rasta experiment.

```
cd ../python
python3 1_testing.py
```

When the testing is completed, the file _img_dataset_testing.csv_ is saved inside the directory _outputs_ and contains the output of the penultimate layer of the network during the testing.

### Step 5: Formatting the image dataset

```
python3 2_format_dataset.py
```

Here, the dataset _img_dataset_testing.csv_ is correctly formatted. The output is the dataset _img_dataset.csv_, which is used for the experiment "Inferring Music And Visual Art Style Evolution via Computational Intelligence" (1).
