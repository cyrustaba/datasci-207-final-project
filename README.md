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
```bash
pip install -r requirements.txt
