---
layout: "post"
title: "Data-Mining-References-Principles"
date: "2017-05-01 19:12"
---

# List Of Basic Data Mining Topics With References and Short Descriptions



## Association Rules

**Notes**



* Basket - This definition comes from the fact, thar AR rulest are most often mined in stores to predict what goods are bought together to recommend stuff to people who buy some goods. In that case, to find these rules, a lot of baskets that are gone through to detect some rules and there comes the term basket. More overally association rules are generated based on some co-occurences of items in groups.
* Support - $\frac{\text{number of baskets with itemset}}{\text{number of all baskets}}$. Itemset can be the size of 1..k. For example how many times beer and diapers occured together in all the baskets.
* $Confidence(a=>b)=\frac{Support(a \cap b)}{Support(a)}$ Itemset occurence together with item B / All itemset occurences. Example: If We want to know the confidence of a rule Beer->Diapers, we divide the number of times beer and diapers occured together with the number beer occured overall.
* $Lift=\frac{Support(a\cap b)}{Support(a)* Support(b)}$ - Exactly like covariance.
* Pass over the data - Iteration over the dataset
* The most computationally expensive part in finding association rules is finding the itemsets.
  * If you have the itemsets with their supports, then to find an association rule for itemset1->some elements with confidence C, you have to find all all itemsets which containt itemset1 + additional elements that have higher support than C*support(itemset1). This is because C<support(S1 and S2)/support(S1)=confidence => C * support(S1) < support(S1 and S2).

**Algorithms**  
_**For all these algorithms I have got an example PDF in the folder Association rules**_

* [Apriori](http://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html/2) : A sort of very easy algorithm to understand. But its recommended to go through an example once to get a clearer understanding of what it actually does.
  * In each pass, we keep in the main memory the itemsets for which we are counting the occurences and the number of occurences of that particular itemset.
  * In first pass the cardinality is 1(Itemset size is 1, so we are counting the occurences of individual items, for examples apple occured in 10 baskets of total 100 baskets)
* PCY -   Usually the second pass of APRIORI, where the itemsets are the size of 2, we have the most work to do, because there are around n**2 itemsets for we which we have to keep count. This is very memory intensive. In PCY algorithm, we try to lessens the number of itemsets we have to keep in main memory during the second pass by using another constraint in 1st phase- an hash function.
  * In pass 1 in addition to counting the number of occurences of each occuring item, we also hash all the occuring **pairs** and count the amount of hash result occurences. If  by the end the pass 1 the occurence of some hash result is smaller than support, we can be 100 % sure that the pair that hashes to this result does not occur more than the support.
* SIMPLE ALGORITHM - divide all the baskets into *n* separate groups and perform apriori on each one of them until pass 2.*DEF Initial support you want to use for the full dataset:S.* Set the level of support in each group K/n * (S) If some itemset does not occur in any of the groups   K/n * (S) times, then it cannot also occur in the full dataset S times, thus it is excluded from the second pass. This method again proposes some additional conditions that itemsets int the size of 2 have to follow to be considered in the 2. phase

**Important  for PCY and SIMPLE ALGORITHM is that these algorithms both focus only reducing the amount of itemsets to consider only in the second pass, because that computationally very expensive. They dont focus anywhere else.**

## PageRank

Added GoThrough Example in folder PageRank
[Best Tutorial](http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html)  
 [With some Algebra Background1](http://www.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture1.html)  
 [Background2, EigenVectors, Eigenvalues]( https://www.khanacademy.org/math/linear-algebra/alternate-bases/eigen-everything/v/linear-algebra-introduction-to-eigenvalues-and-eigenvectors)

### Overall Description

* We consider the internet as a directed graph. Where every web-page stands as a node and and every outgoing hyperlink from that web-page stands as a directed link our graph. _In wired.com we have hyperlink to tesla.com_
* Based on the outgoing links from every node we want to find the importance/ the scale of each web-page in the system.
* Each web-page importance is considered to be total importance of web-pages pointing to it. Each web-page points its importance to other web-pages which it has directed links to. This importance is divided equally between all the web-pages which it points to.
* This forms kind of a recursive system. What we want to find, is each web-pages importance. How to do that?
The solution must come from this overall information we can get up from the structure. More specifically these web-pages have formed together with each other a system of linear equations. Where the formula for each web-page is the sum of the web-pages that point to this web-page.   
* Now we must find a solution to this equation.

### Methods to solve

PAGERANK vector - importance of web-pages in n*1 vector form. Usually marked as r


To do this we are going to reshape this graph in a matrix form, where every row in matrix M is considered as 1 web-page i and every column j consist of nodes going out from web-page j. That all together means that element Mij  the outgoing link from node j to node i.  **This means, that matrix M has to be a column Stochastic Matrix, that means every column sums up to 1** Also every element in 1 column is equal, because importance to other web-pages is shared equally to outgoing web-pages.

#### FUN WITH LINEAR ALGEBRA

Although these methods are not generally use, I still mention them, because they are a good way to introduce the problem we are trying to solve.

* Solve the equation. Too many nodes and links in the internet.
* If we consider r as a pagerank vector, that means the solution to the equation mentioned previously, it follows that rM = r - because every web-page importance is defined by the sum of all other web-page importance. This however results in fact that r is an eigenvector with eigenvalue = 1. Therefore one other way to solve this problem would be to find the eigenvector of M. This is also computationally expensive

#### Power Iteration Method.

##### Idea
In this method, we consider the importance as the probability that we are going to end on each web-page at any time moment. For example, there exist a certain random web-surfer in the web, that endlessly surfs around the internet and we now want to know the probability that this random web-surfer is going to be at some web-page at random moment.

What we can do here is to just simulate this experience and find out the overall amounts of times random surfer visited each web-page among all the times. We don´t actually have to simulate this random surfer, because we can find the probabilities using matrix multiplication.  This is described very well in the "Best tutorial" I previously mentioned.
Overally what we do in this process is:

##### Implementation of idea

* Generate M based on web-pages. Generate r with all values = 1/N, where N = # web-pages.
* perform M*r number of times until r converges.
  * Intuitively, when we perform M*r multiple times, we are in each step moving from each web-page with some probability to another and cumulating these steps together. So in the beginning I might be in equal probability at each web-page, but as the time moves on I actually gonna be at some web-pages more often than other web-pages.
  * You can also think about the movement in a way that the surfer is actually a liquid which is initially poured into N web-pages equally,the total amount  OF LIQUID is 1. With every step this liquid moves out and in from these web pages. At any point of the time more important web-pages have higher **FLOW**  (output is always based on the input, so this logic does not have an issue with some important web-pages having few outlinks ). Like with liquid flow at some point it reaches constant flow, aka it converges.


Thi process overally is considered as a Markov chain. Based on the fact that this chain is irreducible and aperiodic, it will also converge to a stationary distribution.

##### Problems

There are some problems with this approach as well

* Spider Traps - web pages pointing only to inner circles.
  * Random jumps with some probability p. Pretty much we replace M with matrix similiar to it but with random teleports. Formula for that is M=bM + 1-b*U , where U is a matrix with same dimensions as M but, every element in it equals 1/N
* Web-pages that point nowhere - this might result in r = 0 in the end, because theres a column which is non-stochastic. This is important for  
  * Remove those nodes.
  * Follow randome nodes with probability 1 from those nodes
* Matrixes are Super Sparce so a lot of memory is used.
  * We have 1 billion pages. Matrix M would have billion**2 entries. That is a lot to keep in main memory. Because of random jumps, there non zero entries in the matrix
    * To save space, w have to adjust the formula.
    * Initially formula r=(bM+(1-b)* U),  is a matrix with same dimensions as M but, every element in it equals 1/N
    * through some organization r=bMr+[(1-b)* U ] second part is constant. Dont have to keep it in mamory. now we can transform M, sor only nozero links are shown.



I have added Mauro Sozio-s lecture notes with specific examples of overcoming these problems.


# CLUSTERING


Clustering Performs unsupervised machine learning. This is actually a super-interesting topic, which has a lot of relations to Gibbs sampling, Latent Dirichle Allocation Posterior approximation, Bayesian analysis, likelihood maximization, Convergence, Markov Chains, Monte Carlo methods.

Overally the end result is to fit some kind of grouping on the data. The grouping itself usually assume some certain model.

## Methods

### Non-hierachical K-means

*K-means*
1. Choose K  
2. Randomly initiliaze K centroids on data  
3. Assign every datapoint to the group with closest(Euclidean or some other measure) centroid   
4. Find mean in each group.
5. Set the mean in each group to be new centroid.
6. Repeat from 3. step until group/cluster assignement does not change.   

*K-means++*

Differs from previous only in step 2. Everything else is the same. Still very easy.   
In step 2.  Basic Principle is to choose 1. centroid at random, second centroid form dataset with assigning higher probability to elements, that are further away from centroid1. Choose third centroid
with assigning higher probability to datapoints that are further away from centroid 1 and 2.
Principles and math of assignement explained very well on slides in folder Cluster. Also an example executed there.
Some tutorials as well specifying how the initialization works  
http://stats.stackexchange.com/questions/135656/k-means-a-k-a-scalable-k-means -- not exactly same thing, but explains concept.
https://datasciencelab.wordpress.com/2014/01/15/improved-seeding-for-clustering-with-k-means/ -- not too good in explainint


*Why k-means always terminates*

There are ultimately only a finite number of cluster assignments, so if the algorithm ran on forever, you would end up passing through a given assignment more than once. This is impossible because any reasonable K-means algorithm will strictly reduce the error on each step, so you could not possibly come back to the same assignment.


### HIERACHICAL CLUSTERING  
Hierarchicalt Aggromakgtkrnnfjnmaletive
* HIERACHCAL TREES

### Improvements to clusterings

## Evaluation of result

* Entropy http://www.saedsayad.com/decision_tree.htm
* Purity


* SSE of clusters



## Decision Trees

[Explanation](http://www.saedsayad.com/decision_tree.htm)
[Gini index](https://stats.stackexchange.com/questions/77213/computing-the-gini-index)
[Explanation with example](https://github.com/AndresNamm/Classification-Techniques-StepByStep/blob/master/StepByStepHuntAlgorithmDecisionTreeSiniIndex.ipynb)

This is a method of supervised learning. We have some labels, classes which we want to classify elements to from the dataset.

#### Definitions

* Splits - can be binary or more
* Label - classes we are looking to assign the dataset
* Attribute - based on these we perform predictions.
* Node in tree  - If we split based on some attribute, then every different value of that attribute forms a separate node. We calculate Gini index on each node separately based on the class distribution on that node. The
* Gini index - the lower the better.
* Leave - Node which we dont split anymore, because we have reached some stopping condition.

**Attributes can be**

– Nominal (or categorical), have 2 or more
categories, no order.E.g.:country (USA, Spain)
– Ordinal, 2 or more categories but can ordered
or ranked. E.g. T-Shirt size: S,M,L,XL,XXL
– Continuous Ordinal, e.g. temperature, salary

#### Purpose

Assign dataset elements into labels/classes. Labels can be binary or discrete. Cant be Continuous. Then its regression.  

#### Hunts Algorithm


#### Idea

1. Choose attribute with lowest Gini Index.
2. Split dataset based on that attribute.
3. Recurively perform 1,2 until stopping condtion on all subtreest.


**Other criterias for choosing the best attribute for split**

* Entropy
* Information Gain  - Decrease in Entropy  

#### Stopping Conditions on node.  

1. Every element in node  belongs to same target class, eqv to GINI = 0
2. Node becomes empty.


#### How to deal with contiuous attributes

We have to split those attributes into segments. Either binary or something else. It seems logical, that we would like to choose a split so the Gini index would be the lowest

_Idea_

Lets say our Attribute has N different values. We could just try all these different values as split. But that is expensive $O(N^{2})$ Operations. Because we have to recheck every dataelement again.

Instead we

1. Sort record non decreasingly.
2. Start choosing the split attribute based on lowest value in attribute. Set the split higher. Now we have to reassign only elements that are smaller than the next attribute.

This complexity is $O(N log(N))$

_Above mentioned logic can be extended to multinomial splits as well in addition to only binary splits_

#### Dealing with OverFitting e.g. Generalization errors

* Pre Pruning - stop splitting the tree earlier
  * Stop splitting when certain node has less than K (threshhold) elements
  * Stop if further splitting would not improve Gini index or 0 information Gain.
* Post pruning
  * Predict Generalization Error = training error + 0.5 * # leaves If split does not improve this, then PRUNE


# LAZY LEARNERS

## K-NEAREST NEIGBHOURS

Turn dataset into vectors. For every element that we predict, we just find K nearest distance-wize (EXAMPLE: EUCLIDEAN) elements in dataset. We classify new elements into the majority class of those K-nearest elements.

* Curse of dimensionality - use PCA

## NAIVE BAYES METHOD

### Why GOOD

* No need to train.
* Handles missing values Easily. Just skip this calculation
* Fast


Although this theory is based on Bayes theroem. Its simplest to understand this through this formula
$P(C_{i}|X) = P(X_{1}|C_{i})P(X_{2}|C_{i})...P(X_{N}|C_{i})P(C_{i})$
Which intuitively is probabilities of class i having different X attributes * the probabiliy of class appearing.

###Things to consider with this method

If 1 of the condiditional probabilities is 0, then the entire expression becomes 0.

_Fix_
* We can instead of exact probability calculate the Laplace
$P(X_{i}|C)=\frac{N_{ic}+1}{N_{c}+\text{# classes}}$


### How to deal with Continuous cases

* Discretazation, a.k.a finding the best split again like with decision trees.
* Model based - assume model DISTRIBUTION on data and calculate the probability based on model
  * Normal DISTRIBUTION - heights for example
  * Binomial DISTRIBUTION word occurences
# Classifier Evaluation

_References_

* [Confusion Matrix/Cost Matrix Based Methods](http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)    
* [Another Tutorial about same thing ](http://www2.cs.uregina.ca/~dbd/cs831/notes/confusion_matrix/confusion_matrix.html)





_Metrics to test classifier goodness_


Im gonna use A very simple system 

+ TT - predict true, is true - true positive TP
+ FF - predict false, is false - true negative TN
+ FT - predict false, is true - false negative FN
+ TF - predict true, is false - false positive FP





$$
Confusion Matrix = \begin{bmatrix}   & \text{PREDICT NO} & \text{PREDICT YES} \\ \text{ACTUAL NO} & TN & FP \\ \text{ACTUAL YES} & FN & TP  \end{bmatrix}  
$$

$Accuracy=\frac{TT+FF}{TT+FF+FT+TF}$  
$Precision=\frac{TT}{TT+TF}$ When I predict yes, how often am I correct.  
$Recall=\frac{TT}{TT+FT}$ How many positive examples I predict accuratly.
$\text{F-value}=\frac{Precision*Recall}{Precision + Recall} * \text{Harmonic Mean }$
$\text{Harmonic Mean}(e_{1},e_{2},e_{3})=\frac{1}{\frac{1}{e_{1}}+\frac{1}{e_{2}}+\frac{1}{e_{2}}}$


_Methods to evaluate Classification Accuracy_

* Split data into $\alpha*data$ training and $(1-\alpha)* data$ test set.  - Wastes some data
* Random subsampling:

  1. Sample sample1, sample2
  2. Train on sample1
  3. Test on sample2  
  4. Repeat
* Cross Validation K fold
  1. Partition data into K disjoint sets
  2. Train on k-1 sets, test on remaining one. Repeat k times, each subset being used exactly once. Compute average of results.
* Bootstrapping
  1. Sample data
  2. Replace it
