# Naive Bayes Classifier for Sentiment Analysis

Using Naive Bayes Classifier to detect hate speech in tweets using the dataset provided by Analytics Vidhya. A tweet is considered to contain hate speech if it has a racist or sexist sentiment associated with it. 
<br>
Label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist.
<br>
Link to the dataset [here](https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech).


Naive Bayes classifier is a collection of machine learning models that are used for classification problems. The algorithm is based on Bayes’ Theorem. 
The simple form of the calculation for Bayes’ Theorem is:
	P(A|B) = P(B|A) * P(A) / P(B)

The probability that we are interested in calculating; P(A|B), is called the posterior probability and the marginal probability of the event; P(A) is called the prior.

Consider x as predictors and y as outcome.
	P(yi | x1, x2, …, xn) = P(x1, x2, …, xn | yi) * P(yi) / P(x1, x2, …, xn)

The prior P(yi) is easy to estimate from a dataset, but the conditional probability of the observation based on the class P(x1, x2, …, xn |yi) is not feasible unless the number of examples is extraordinarily large, e.g. large enough to effectively estimate the probability distribution for all different possible combinations of values.

The solution to using Bayes’ Theorem for a conditional probability classification model is to simplify the calculation.

The Bayes Theorem assumes that each input variable is dependent upon all other variables. This is a cause of complexity in the calculation. We can remove this assumption and consider each input variable as being independent from each other.

First, the denominator is removed from the calculation as it is a constant used in calculating the conditional probability of each class for a given instance and has the effect of normalizing the result.
(For all entries in the dataset, the denominator does not change, it remain static. Therefore, the denominator can be removed and a proportionality can be introduced.)
	P(yi | x1, x2, …, xn) = P(x1, x2, …, xn | yi) * P(yi)

Next, the conditional probability of all variables given the class label is changed into separate conditional probabilities of each variable value given the class label. These independent conditional variables are then multiplied together.
	P(yi | x1, x2, …, xn) = P(x1|yi) * P(x2|yi) * … P(xn|yi) * P(yi)

This calculation can be performed for each of the class labels, and the label with the largest probability can be selected as the classification for the given instance. This decision rule is referred to as the maximum a posteriori (MAP) decision rule.

TYPES 
There are 3 distinct Naive Bases classifier algorithms. As an example, in a case of sentiment analysis, we’ll be using these algorithms when the predictors are: 
- Bernoulli NB: Independent booleans (binary variables), e.g. if a word occurs in the text or not.
- Multinomial NB: Count of word occurence.
- Gaussian NB: TF-IDF vectors.
