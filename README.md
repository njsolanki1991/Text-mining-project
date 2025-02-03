**Text Mining Project - Author Verification**
**Project Group - Team 11**

This repository contains the following python files(.py and .ipynb) 

***1. ExtractAndSplitDataset.py***

This file extracts complete Pan20 dataset zip files and divides the dataset in 90:10 ratio for training and test set for dissimilarity. The resulting files are stores as pairs.jsonl for text data and truth.jsonl for groundtruth for each training and test dataset respectively.

***2. PackageInstallationAndExtractionAndSplittingOfDataset.ipynb***

This file consists of libraries that are needed to be installed for dissimilarity method. And Also the call to the ExtractAndSplitDataset.py is done. Make sure that pan-20 zip file is present in the required location before running this notebook.

***3. pan20_verif_evaluator.py***

This modified evaluation file is from PAN20 challenge for author verification. This file will be used to evaluate the performance of our method.

Check the following for source code:
https://github.com/pan-webis-de/pan-code/blob/master/clef20/authorship-verification/pan20_verif_evaluator.py

***4. PAN2020_DissimilarityAlgorithm.ipynb***

In this file, the dissimilarity method with POS-Tag features with 100 profile length for Verb, Noun, Pronoun, Adjective is implemented. Also features for Word unigram, Word bigram, Word trigram are implemented for both 100 profile length and 200 profilelength. And also features for Character 4-gram, Character 5-gram, Character 6-gram, Character 7-gram, Character 8-gram are implemented for both 100 profile length and 200 profilelength.
