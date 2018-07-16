<img src="https://i.imgur.com/kYL058E.png" height="150px">

Scikit-learn inspired model finetuning for natural language processing.

`Finetune` ships with a pre-trained language model
from ["Improving Language Understanding by Generative Pre-Training"](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
and builds off the [OpenAI/finetune-language-model repository](https://github.com/openai/finetune-transformer-lm).

Finetune Quickstart Guide
=========================

Finetuning the base language model is as easy as calling `Classifier.fit`:

```python3
from sklearn.model_selection import train_test_split
from finetune import Classifier
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank

df = StanfordSentimentTreebank(nrows=100).dataframe                 # load 100 rows from example dataset
model = Classifier()                                                # load base model
trainX, testX, trainY, testY = train_test_split(df.Text, df.Target) # split data in train and test sets
model.fit(trainX, trainY)                                           # finetune base model on custom data
predictions = model.predict(testX)                                  # [{'class_1': 0.23, 'class_2': 0.54, ..}, ..]
model.save('./saved-model')                                         # serialize the model to disk
```

Easily reload saved models from disk by using `LanguageModelClassifier.load`:

```
model = Classifier.load('./saved-model')
predictions = model.predict(testX)
```


Installation
============
Finetune can be installed directly from PyPI by using `pip`

```
pip3 install finetune
```

or installed directly from source:

```bash
git clone https://github.com/IndicoDataSolutions/finetune
cd finetune
python3 setup.py develop
python3 -m spacy download en
```

You can optionally run the provided test suite to ensure installation completed successfully.

```bash
pip3 install nose
nosetests
```
