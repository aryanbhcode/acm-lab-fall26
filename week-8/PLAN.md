# Week 8 extension plan

## Hypothesis

I think LIME needs a lot of samples to find the words a model really uses. With only 50 samples I expect it to miss a lot of them. As I add more samples the recall should go up and then level off, probably somewhere around 1000 to 2500 samples. I also think long posts will need more samples than short ones, because there are more words to sort through.

## Method

I will take my Week 7 recall code and change just one thing, which is how many samples LIME uses for each explanation. I will try 50, 100, 250, 500, 1000, 2500 and 5000. I will run 60 posts for each model, repeat it with 3 different random seeds, and compare LIME against the Random and Greedy baselines. The code is in week-8/extension.py.

## Data and models

I am using the same data as Week 7, which is the 20 Newsgroups posts about atheism and christianity. The two models are a sparse logistic regression that only uses 10 words, and a decision tree with a depth of 10.

## Success criteria

The extension works if I get a clear curve of recall against sample size for both models, and I can say about how many samples LIME needs to reach 90 percent recall and match Greedy. It also works if long posts really do get lower recall than short posts when the sample size is small. If recall is already close to 100 percent at 50 samples, then my hypothesis was wrong, and I will still count that as a real result.
