# classify_language

This project implements different models to classify the language a text is written in.
The training data contains 1000 lines for each of 10 languages.
Models contain a naive approach computing probabilities based on word counts in the training data and several machine / deep learning approaches. Among them is a character level CNN and LSTM and bag-of-words model.

## Data

The training data used is taken from:
https://www.kaggle.com/zarajamshaid/language-identification-datasst?select=dataset.csv,
which is an excerpt from the Wikipedia Language Identification Dataset (https://arxiv.org/abs/1801.07779).

It is further processed by selecting the following 10 languages with a latin based alphabet:
- Estonian
- Swedish
- Dutch
- Turkish
- Latin
- Portugese
- French
- Spanish
- Romanian
- English

All characters are transformed to ASCII lowercase. Punctiation, extra white spaces and numbers are removed. This preprocessing is done in `src.data`.
For all trainings a train/validation split of 80/20 is used.

## Models

The language of a text may most easily be predicted by creating a dictionary of words and their language. Upon inference a simple vote of words should then provide a good estimate. This is done in the naive classifier.
For any machine learning model it appears reasonable to begin with a character level approach due to the large vocabulary of multiple languages and the much smaller number of latin characters.
Language specific patterns in character sequences should be easily identifiable. It should be a much simpler task than e.g. sentiment classification. Thus, there is a character level CNN and an LSTM.
On the other hand when trained specifically for language classification word embeddings may be very easily separable. A linear classifier could be enough. This is tried in the BOW model. 

### Naive Classifier

This simple model iterates over the training data and for each unique word generates a histogram of how often it appears in all languages. At inference time it iterates over the string to be predicted and sums up the histograms of known words, i.e. words that were included in the training data. Unknown words are simply skipped. The summed histograms are normalized and give a probabilistic estimate which language the given string is written in.
It is fast to train and works very accurately.

### Character CNN

To be processed by a CNN characters are represented as one-hot-vectors. Several 1D-convolutional layers are topped by two fully-connected layers to produce an estimate.
The architecture is motivated by (https://arxiv.org/pdf/1509.01626.pdf).

### LSTM

A single LSTM layer receives as input one-hot-vectors of character sequences.
A fully connected layer then produces an estimate from either the mean of all hidden states or the last hidden state only.

### BOW

A sparse embedding layer is topped by a single linear classification layer.
All word embeddings of a sequence are averaged before being processed by the linear layer. Thus the objective for the embedding layer is to form linearly seperable clusters of words in the embedding space.
To prevent out-of-vocabulary issues the vocabulary for the embedding layer is initialized from the entire data.
But of course embeddings for words not contained in the training set are never optimized for and can only contribute noise to an estimation.

## Results

## Discussion
removing punctiation makes is difficult to identify citations.
BOW not robust against out-of-vocab isssues whereas naive and char based models are.

## Usage and Implementation