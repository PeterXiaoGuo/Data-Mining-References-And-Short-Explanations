---
layout: "post"
title: "Data-Mining-References-Principles"
date: "2017-05-01 19:12"
---

# List Of Basic Data Mining Topics With References and Short Descriptions



## Association Rules

**Definitions**

* Basket - This definition comes from the fact, thar AR rulest are most often mined in stores to predict what goods are bought together to recommend stuff to people who buy some goods. In that case, to find these rules, a lot of baskets that are gone through to detect some rules and there comes the term basket. More overally association rules are generated based on some co-occurences of items in groups.
* Support - _Itemset Occurence in basket/# of all baskets_. Itemset can be the size of 1..k. For example how many times beer and diapers occured together in all the baskets.
* Confidence - Itemset occurence together with item B / All itemset occurences. Example: If We want to know the confidence of a rule Beer->Diapers, we divide the number of times beer and diapers occured together with the number beer occured overall.
* Pass over the data - Iteration over the dataset
* The most computationally expensive part in finding association rules is finding the itemsets.

**Algorithms**  
_**For all these algorithms I have got an example PDF in the folder Association rules**_

* [Apriori](http://www.kdnuggets.com/2016/04/association-rules-apriori-algorithm-tutorial.html/2) : A sort of very easy algorithm to understand. But its recommended to go through an example once to get a clearer understanding of what it actually does.
  * In each pass, we keep in the main memory the itemsets for which we are counting the occurences and the number of occurences of that particular itemset.
  * In first pass the cardinality is 1(Itemset size is 1, so we are counting the occurences of individual items, for examples apple occured in 10 baskets of total 100 baskets)
* PCY -   Usually the second pass of APRIORI, where the itemsets are the size of 2, we have the most work to do, because there are around n**2 itemsets for we which we have to keep count. This is very memory intensive. In PCY algorithm, we try to lessens the number of itemsets we have to keep in main memory during the second pass by using another constraint in 1st phase- an hash function.
  * In pass 1 in addition to counting the number of occurences of each occuring item, we also hash all the occuring **pairs** and count the amount of hash result occurences. If  by the end the pass 1 the occurence of some hash result is smaller than support, we can be 100 % sure that the pair that hashes to this result does not occur more than the support.
* SIMPLE ALGORITHM - divide all the baskets into *n* separate groups and perform apriori on each one of them until pass 2.*DEF Initial support you want to use for the full dataset:S.* Set the level of support in each group K/n * (S) If some itemset does not occur in any of the groups   K/n * (S) times, then it cannot also occur in the full dataset S times, thus it is excluded from the second pass. This method again proposes some additional conditions that itemsets int the size of 2 have to follow to be considered in the 2. phase

**Important  for PCY and SIMPLE ALGORITHM is that these algorithms both focus only reducing the amount of itemsets to consider only in the second pass, because that computationally very expensive. They dont focus anywhere else.**
