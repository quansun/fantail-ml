### Description

Fantail is a collection of machine learning algorithms for ranking prediction, multi-target regression, label ranking and metalearning related data mining tasks. The algorithms can be called from your own Java code. It is also well-suited for developing new algorithms. Fantail is a multi-target learning extension to WEKA, and is at the early development stage. New algorithms and tools will be added to the library gradually.

A key difference between Fantail and another popular preference learning package WEKA-LR is: Fantail uses the rank vector format (similar to the multi-target regression setting) rather than the order/explicit preference vector format. So in Fantail, label ranking is treated as a special case of the multi-target regression problem. The advantage of the Fantail approach is that both multi-target and label ranking algorithms can be used and tested under a unified framework.

### Usage

See fantail.examples.LabelRankingSingleAlgoExample01.java for an example

### Benchmark Datasets

/datasets/iris_.arff (an example dataset showing the data format used by Fantail) 

A collection of 26 label ranking datasets can be downloaded from /datasets

### Algorithms (8)

* AverageRanking (a baseline ranker)
* RankingWithkNN (based on a nearest neighbour algorithm)
* RankingWithBinaryPCT (based on predictive clustering tree for ranking)
* RankingByPairwiseComparison
* BinaryART (approximate ranking tree)
* ARTForests (approximate ranking tree forests)
* Label Ranking Tree (WEKA-LR's LRT, note: this algorithm has been removed from version 1-1-3)
* RankingViaRegression (multiple single-target regression)

### Evaluation Metrics

* Spearman's rank correlation coefficient
* Kendall's Tau
* MAE
* RMSE

### TODOs (major features)

* Curds and Whey Multivariate Responses
* Constraint Classification
* Bagging
* Boosting
* MetaRule Generator
* NDCG@X
* GUI/Visualisation
* 2D/3D permutation polytopes for rank data
* Experimenter

### Citing Fantail

If you want to refer to Fantail in a publication, please cite the following paper: 

Quan Sun and Bernhard Pfahringer. Pairwise Meta-Rules for Better Meta-Learning-Based Algorithm Ranking. Machine Learning, 93(1):141-161, Springer US, 2013, DOI: 10.1007/s10994-013-5387-y

Many TODOs, so please send me an Email if you would like to contribute to Fantail!
Quan Sun quan.sun.nz@gmail.com 
10/2013 
