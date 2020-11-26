# ContrastRetrieval



## Downloading data and pretrained models

### Data

1. Download Yelp data: https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports and place files in datasets/AUS_dataset/raw/
Run script to pre-process script and create train, val, test splits:
bash scripts/preprocess_data.sh
Download subword tokenizer built on Yelp and place in datasets/yelp_dataset/processed/: link