# KGDB-DDI Repository

## How to Run KGDB-DDI
Execute one of the following commands based on your target dataset:

1. For the ​**DrugBank** dataset:
```bash
python predict.py --batch_size 64 --dataset Drugbank --epochs 100 --lr 0.0001 --gpu 0 --p 0.1
```

2. For the ​**KEGG** dataset:
```bash
python predict.py --batch_size 8 --dataset KEGG_DRUG --epochs 100 --lr 0.00005 --gpu 0 --p 0.1
```
