# DATASCI 207 Final Project
## Predicting Music Genre from Mel Spectrograms

### Team
- Caroline Nealon
- Hussain Zaidi
- Matthew Thomas
- Cyrus Tabatabai

### Overview
Can we accurately predict a track's genre or popularity score based purely on its underlying audio features? We convert raw audio waveforms into mel spectrograms using librosa, then feed them into image classification models.

### Pipeline
1. Audio → Mel Spectrogram (librosa)
2. Spectrogram images → CNN / Transfer Learning models
3. Model interpretability via SHAP

### Dataset
[FMA: Free Music Archive](https://github.com/mdeff/fma)

### Setup
To set up the pipeline, first sync dependencies:
```bash
pip install -r requirements.txt
uv sync
```
Navigate to `notebooks/01_SetupDataFolder.ipynb` and follow instructions for getting the `/data` folder set up.

The data must be downloaded from an independent github repository and places in the data folder for the setup file to run properly.
The purpose of this is to have the data stored locally on each machine, rather than in the github repository because of storage constraints

