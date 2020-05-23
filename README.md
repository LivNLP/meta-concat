# PIP-concat
Creates meta embedding via unsupervised PIP loss minimisation

This branch hold the code base for creating co-occurrence matrices directly from a given text corpus. However, for larger corpora 
or for vocabularies (greater than 15K) you will run into memory issues with this branch and it is better to use the cooc branch for
loading pre-computed co-occurrences matrices in that case.
