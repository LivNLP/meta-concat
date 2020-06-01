# Meta-embedding by weighted concatenation of sources
This project implements the meta-embedding method where source embeddings are weighted prior to concatenating row-wise. 
The concatenation weights are learnt such that the pairwise inner-product (PIP) loss is minimised according bias and variance terms, that
involve the spectra of the ideal meta-embedding signal matrix and signal matrices for the individual source embeddings.

## Requirements
- Python 3
- PyTorch
- tabulate
- pandas
- [repseval](https://github.com/Bollegala/repseval)
- [senteval](https://github.com/facebookresearch/SentEval)

## Reproducing results
To compute the dimension-weighted and source-weighted meta-embeddings do the following, where corpus_file is a tokenised text file
from which we will be computing the signal matrices for the source embeddings, output_file is a file to which we will be writing
various performance metrics and alpha (in range [0,1]) is the parameter that magnifies the singular values during decomposition
(see paper for further details)
```
    python concat.py corpus_file output_file alpha
```

If you want to reproduce the results for baseline methods (SVD, AVG) and previously proposed meta-embedding learning methods (LLE and 1toN),
then use the baselines.py as follows.
```
    python baselines.py
```


