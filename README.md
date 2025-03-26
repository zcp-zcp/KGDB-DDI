# KGDB-DDI Repository

## Model Note
Due to the large size of ​**RoBERTa_DDI**, we have not uploaded the model file. Instead, we provide pre-generated embeddings for drug background data using RoBERTa_DDI, which are stored in the `Data` folder.

## How to Run KGDB-DDI
Execute one of the following commands based on your target dataset:

1. For the ​**DrugBank** dataset:
```bash
python predict.py --batch_size 64 --dataset Drugbank --epochs 100 --lr 0.0001 --gpu 0 --p 0.1

1. For the ​**KEGG** dataset:
```bash
python predict.py --batch_size 8 --dataset KEGG_DRUG --epochs 100 --lr 0.00005 --gpu 0 --p 0.1

If you require the full ​RoBERTa_DDI model , please contact us via email.
