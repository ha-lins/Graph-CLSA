## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)

Then, you need to create a directory for recoreding finetuned results to avoid errors:

```
mkdir logs
```

## Training & Evaluation

### For SimCLR (i.e., GraphCL)
```
./go.sh $GPU_ID $DATASET_NAME $AUGMENTATION
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.

### For CLSA
```
./clsa.sh $GPU_ID $DATASET_NAME $AUGMENTATION $STRO_AUGMENTATION
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.
``$STRO_AUGMENTATION`` is {stro_dnodes, stro_subgraph}. Such as: 

`./clsa.sh 2 COLLAB dnodes stro_dnodes`.

### For BYOL
```
CUDA_VISIBLE_DEVICES=2 python gbyol.py --DS COLLAB --lr 0.01  --aug dnodes --stro_aug stro_dnodes
```
Or
```
./byol.sh $GPU_ID $DATASET_NAME $AUGMENTATION $STRO_AUGMENTATION
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.
``$STRO_AUGMENTATION`` is {stro_dnodes, stro_subgraph}.



## Acknowledgements

1. The backbone implementation is reference to: https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.
2. The BYOL implementation is reference to: https://github.com/lucidrains/byol-pytorch