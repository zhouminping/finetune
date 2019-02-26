import os
import logging
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from finetune import MultiTask
from finetune.datasets.stanford_sentiment_treebank import StanfordSentimentTreebank

logging.basicConfig(level=logging.DEBUG)

SST_FILENAME = "SST-binary.csv"
DATA_PATH = os.path.join('Data', 'Classify', SST_FILENAME)
CHECKSUM = "02136b7176f44ff8bec6db2665fc769a"

if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = StanfordSentimentTreebank(nrows=100).dataframe
    model = MultiTask(tasks={
        "sst1": "classification",
        "sst2": "classification",
    }, debugging_logs=True)
    trainX, testX, trainY, testY = train_test_split(
        dataset.Text.values, dataset.Target.values, test_size=0.3,  random_state=42
    )
    model.fit(
        {
            "sst1": trainX,
            "sst2": trainX,
        },
        {
            "sst1": trainY,
            "sst2": trainY,
        }
    )
    accuracy = np.mean(model.predict(testX) == testY)
    print('Test Accuracy: {:0.2f}'.format(accuracy))
