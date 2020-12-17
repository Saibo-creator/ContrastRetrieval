# ContrastRetrieval



## Downloading data and pretrained models

### Data

1. Download Yelp data: https://archive.ics.uci.edu/ml/datasets/Legal+Case+Reports and place files in datasets/AUS_dataset/raw/
Run script to pre-process script and create train, val, test splits:
bash scripts/preprocess_data.sh
Download subword tokenizer built on Yelp and place in datasets/yelp_dataset/processed/: link
   

In validation split:
- 350 cases
- 1750 catchphrases
In test split:
- 350 cases
- 1750 catchphrases
In train split:
- 2802 cases
- 14010 catchphrases

# Train:

## Model
Encoder: BERT_BASE(freezed)
function: 2 Dense Layer

## Loss
Loss Contrastive loss: Visual-Semantic Embeddings with Hard Negatives 

diaganal elements are positive

#Retrival:

Cosine Similarity
S(f(x),y)

x= sentence(encoded by BERT, f is two dense layer)
y = catchphrases



## Metric
Recall@k
Precision@k
RecallPrecision@k
NDCG@k

## Bert-base,

R@1 : 0.001   P@1 : 0.003   RP@1 : 0.003   NDCG@1 : 0.003
R@3 : 0.002   P@3 : 0.003   RP@3 : 0.003   NDCG@3 : 0.003
R@5 : 0.003   P@5 : 0.003   RP@5 : 0.003   NDCG@5 : 0.003
R@7 : 0.004   P@7 : 0.003   RP@7 : 0.004   NDCG@7 : 0.004
R@9 : 0.005   P@9 : 0.003   RP@9 : 0.005   NDCG@9 : 0.004


After 1 epoch:

R@1 : 0.001   P@1 : 0.003   RP@1 : 0.003   NDCG@1 : 0.003
R@3 : 0.002   P@3 : 0.003   RP@3 : 0.003   NDCG@3 : 0.003
R@5 : 0.003   P@5 : 0.003   RP@5 : 0.003   NDCG@5 : 0.003
R@7 : 0.004   P@7 : 0.003   RP@7 : 0.004   NDCG@7 : 0.004
R@9 : 0.005   P@9 : 0.003   RP@9 : 0.005   NDCG@9 : 0.004