# MDGFM
<h1 align="center"> Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment </a></h2>



## Setup Environment

- python 3.9.20
- pytorch 1.10.1+cu113
- torch_cluster 1.6.0
- torch_geometric 2.1.0
- torch_scatter 2.0.9
- torch_sparse 0.6.13
- torch_spline_conv 1.2.1
- dgl 0.9.1
- cuda 11.3
- pyG 2.1.0
  
## Running experiments

'preprompt.py' is the code for the pre training phase, 'downprompt.py' is downstream code and 'MDGFM.py' is the entire code of our model. 

### For different datasets, please run the following code：
#### Cora:
> one-shot
```
python runexp.py --dataset Cora --drop_percent 0.5 --lr 0.0075 --downstreamlr 0.001 --epochs 60 --shot_num 1
```
> few-shot
```
python runexp.py --dataset Cora --drop_percent 0.5 --lr 0.0075 --downstreamlr 0.001 --epochs 60 --shot_num 5
```
#### Citeseer:
> one-shot
```
python runexp.py --dataset Citeseer --drop_percent 0.5 --lr 0.001 --downstreamlr 0.001 --epochs 60 --shot_num 1
```
> few-shot
```
python runexp.py --dataset Citeseer --drop_percent 0.5 --lr 0.001 --downstreamlr 0.001 --epochs 60 --shot_num 5
```
#### Pubmed:
> one-shot
```
python runexp.py --dataset Pubmed --drop_percent 0.5 --lr 0.0001 --downstreamlr 0.0014 --epochs 60 --shot_num 1
```
> few-shot
```
python runexp.py --dataset Pubmed --drop_percent 0.5 --lr 0.0001 --downstreamlr 0.0014 --epochs 60 --shot_num 5
```
#### Cornell:
> one-shot
```
python runexp.py --dataset Cornell --drop_percent 0.5 --lr 0.02 --downstreamlr 0.0003 --epochs 100 --shot_num 1
```
> few-shot
```
python runexp.py --dataset Cornell --drop_percent 0.5 --lr 0.02 --downstreamlr 0.0003 --epochs 100 --shot_num 3
```
#### Chameleon:
> one-shot
```
python 0runexp.py --dataset Chameleon --drop_percent 0.5 --lr 0.02 --downstreamlr 0.02 --epochs 100 --shot_num 1
```
> few-shot
```
python runexp.py --dataset Chameleon --drop_percent 0.5 --lr 0.02 --downstreamlr 0.02 --epochs 100 --shot_num 5
```
#### Squirrel:
> one-shot
```
python runexp.py --dataset Squirrel --drop_percent 0.5 --lr 0.01 --downstreamlr 0.0003 --epochs 100 --shot_num 1
```
> few-shot
```
python runexp.py --dataset Squirrel --drop_percent 0.5 --lr 0.01 --downstreamlr 0.0003 --epochs 100 --shot_num 3
```
#### Penn94:
Extensive experiments on Penn94 can be conducted using *MDGFM_penn.py*, where code is totally identical to *MDGFM.py* except for an increase in the number of source domains from 5 to 6.
> k-shot(k=1,5,10,50,100,500)
```
python MDGFM_penn.py --shot_num k #Please modify the k value by yourself
```
## Data
All datasets could be publicly downloaded, few-shot samples are divided following previous works.

