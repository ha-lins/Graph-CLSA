## Dependencies

* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric#installation)

Then, you need to create a directory for recoreding finetuned results to avoid errors:

```
mkdir logs
```

## Training & Evaluation

For Simclr(GraphCL)
```
./go.sh $GPU_ID $DATASET_NAME $AUGMENTATION
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.

For CLSA:
```
./clsa.sh $GPU_ID $DATASET_NAME $AUGMENTATION $STRO_AUGMENTATION
```

```$DATASET_NAME``` is the dataset name (please refer to https://chrsmrrs.github.io/datasets/docs/datasets/), ```$GPU_ID``` is the lanched GPU ID and ```$AUGMENTATION``` could be ```random2, random3, random4``` that sampling from {NodeDrop, Subgraph}, {NodeDrop, Subgraph, EdgePert} and {NodeDrop, Subgraph, EdgePert, AttrMask}, seperately.
``$STRO_AUGMENTATION`` is {stro_dnodes, stro_subgraph}
## Acknowledgements

The backbone implementation is reference to https://github.com/fanyun-sun/InfoGraph/tree/master/unsupervised.
