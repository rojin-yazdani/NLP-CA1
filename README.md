# N-grams and Perplexity Measure

## Introduction
This exercise was completed in preparation for the Natural Language Processing (NLP) course at [Tehran University](https://ut.ac.ir/en) in March 2020.

The purpose of this exercise is to become familiar with the topics of n-grams and perplexity measure in Persian language. During this exercise, it is expected that you will create relevant language models in Persian according to the n-grams, make your predictions using the perplexity measure, and finally report the evaluation of the results using the F1 metric.


### Dataset
Persian dataset was created from the Hamshahri newspaper news between 1996 and 2006 using web crawlers. Due to the large amount of original data, a sample of it has been created and provided for this exercise.

The number of 2,381 news texts in 6 classes (technology, sports, social, political, finance, and culture) has been placed in the train data section. Also, 600 news texts without class have been included in the test data section

### Exercise
First, divide the dataset into two parts: training (80%) and validation (20%).
Create character- and word-based, unigram and bigram language models from train data and existing libraries for each class.

For each of the news texts in the validation data, determine the desired class according to the minimum amount of Perplexity measure obtained from the relevant language model.

[Exercise notebook](NLP-CA1-Project.ipynb)
