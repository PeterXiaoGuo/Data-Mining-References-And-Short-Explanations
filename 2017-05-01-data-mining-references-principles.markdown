---
layout: "post"
title: "Data-Mining-References-Principles"
date: "2017-05-01 19:12"
---

# List Of Basic Data Mining Topics With References and Short Descriptions



## Association Rules

**Notes**

* Basket - This definition comes from the fact, thar AR rulest are most often mined in stores to predict what goods are bought together to recommend stuff to people who buy some goods. In that case, to find these rules, a lot of baskets that are gone through to detect some rules and there comes the term basket. More overally association rules are generated based on some co-occurences of items in groups.
* Support - _Itemset Occurence in basket/# of all baskets_. Itemset can be the size of 1..k. For example how many times beer and diapers occured together in all the baskets.
* Confidence - Itemset occurence together with item B / All itemset occurences. Example: If We want to know the confidence of a rule Beer->Diapers, we divide the number of times beer and diapers occured together with the number beer occured overall.
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

What we can do here is to just simulate this experience and find out the overall amounts of times random surfer visited each web-page among all the times. We don't actually have to simulate this random surfer, because we can find the probabilities using matrix multiplication.  This is described very well in the "Best tutorial" I previously mentioned.
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
