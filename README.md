# classify_language

This project implements different models to classify the language a text is written in.
The training data contains 1000 lines for each of 10 languages.
Models contain a naive approach computing probabilities based on word counts in the training data and several machine / deep learning approaches. Among them is a character level CNN and LSTM and a bag-of-words model.

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
Language specific patterns in character sequences should be easily identifiable. It should be a much simpler task than e.g. sentiment classification. There is a character level CNN, an LSTM and an attention based model.
On the other hand when trained specifically for language classification word embeddings may be very easily separable. A linear classifier could be enough. This is tried in the BOW model. 

### Naive Classifier

This simple model iterates over the training data and for each unique word generates a histogram of how often it appears in all languages. At inference time it iterates over the string to be predicted and sums up the histograms of known words, i.e. words that were included in the training data. Unknown words are simply skipped. The summed histograms are normalized and give a probabilistic estimate which language the given string is written in.
It is fast to train and works very accurately.

### Character CNN

To be processed by a CNN characters are represented as one-hot-vectors. Several 1D-convolutional layers are topped by two fully-connected layers to produce an estimate.
The architecture is motivated by (https://arxiv.org/pdf/1509.01626.pdf).

### LSTM

Two stacked LSTM layer receives as input one-hot-vectors of character sequences.
Two connected layers then produce an estimate from either the mean of all hidden states or the last hidden state only.

### Attention

This architecture is inspired by the Transformer (https://arxiv.org/abs/1706.03762).
It consists of a multi-head attention layer followed by two fully conntected layers. However, rather than producing a sequential output with the same length as the input, the fully connected layers pool the input to compute a classification output.

### BOW

A sparse embedding layer is topped by a single linear classification layer.
All word embeddings of a sequence are averaged before being processed by the linear layer. Thus the objective for the embedding layer is to form linearly seperable clusters of words in the embedding space.
To prevent out-of-vocabulary issues the vocabulary for the embedding layer is initialized from the entire data.
But of course embeddings for words not contained in the training set are never optimized for and can only contribute noise to an estimation.

## Results

The following table shows the mean accuracies of all models:

| Model     | CNN       | LSTM      | Naive     | Attention | BOW       |
| ----:     | :---:     | :---:     | :---:     | :---:     | :---:     |
| **Acc**   | 88 ± 5    | 95 ± 4    | 98 ± 2    | 75 ± 12   | 97 ± 3    | 

All values are in percent, errors are standard deviations over the ten languages.
Despite its simplicity the naive approach is the best performing followed by the bag-of-words model. Among the character based models the LSTM performs best.
Full confusion matrices are included in `results/`. It also includes failure cases for all models.

For the deep learning based approaches `results/` includes plots of accuracy vs. epoch for both training and validation data. It reveils that the BOW-model is the slowest to learn in the beginning but ultimately exceeds the character based sequential models. The Attention based model strongly overfits the training data, quickly reaching a training accuracy of 100% but never exceeding 80% on the validation data. The LSTM learns faster and overfits less than the CNN.

## Discussion

All character level approaches are intrinsically robust against words not included in the training data. So is the naive approach because it simply skips unknown words. However, it runs into trouble if no word in a given text is know. The bag-of words model is not robust against out-of-vocabulary issues. But a simple pre-processing step could bypass this.

Some failure cases seem to be due to quotations which are difficult to recognize when punctuation is removed. 

Arguably, unordered word level approaches seem to outperform sequential character level ones. Technically, the sequential models could be combined with an embedding layer to operate on a word level. But given the good performance of the unstructured word level models this does not seem to be necessary.

Overfitting could be further reduced by augmenting the start position and length of an input line and by randomly masking out words or characters from the input.

The naive approach works remarkably well even with few training examples and fitting it is much faster than training the deep learning based models.

## Usage and Implementation

The code is written in Python 3.8. All dependencies are included in `requirements.txt` and can be installed by `$ pip install -r requirements.txt`.

Trainings are defined by a `config.yml` as included in `configs/`. All deep learning based models are trained by configuring and running the `train` script from the top level:
```
$ python -m src.train --config <config.yml>
```
The training procedure is defined in `deep_lang_classifier`, PyTorch models are collected in `models/`.

The naive classifier is fitted by running:
```
$ python -m src.naive_lang_classifier --config <config.yml>
```

