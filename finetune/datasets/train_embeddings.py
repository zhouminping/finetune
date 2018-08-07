import os

import logging
from pathlib import Path

from finetune import MultifieldClassifier, Comparison
from finetune.datasets import Dataset, comparison_download
import tensorflow as tf
from finetune.utils import find_trainable_variables

from sklearn.model_selection import train_test_split
import numpy as np

logging.basicConfig(level=logging.DEBUG)

QUORA_SIMILARITY = "Quora.csv"
DATA_PATH = os.path.join('Data', 'Similarity', QUORA_SIMILARITY)
CHECKSUM = "6f2642eec22c1f0f3fc2fb2dfb9dfc38899d7c58bfc9f4c7e4d1a7e3cea15a29"


class QuoraDuplicate(Dataset):
    def __init__(self, filename=None, **kwargs):
        super().__init__(filename=(filename or DATA_PATH), **kwargs)

    @property
    def md5(self):
        return CHECKSUM

    def download(self):
        """
        Download quora duplicate questions dataset.
        """
        path = Path(self.filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison_download(
            url="http://qim.ec.quoracdn.net/quora_duplicate_questions.tsv",
            text_column1="question1",
            text_column2="question2",
            target_column="is_duplicate",
            filename=QUORA_SIMILARITY
        )


if __name__ == "__main__":
    # Train and evaluate on SST
    dataset = QuoraDuplicate(nrows=5000).dataframe
    model = MultifieldClassifier(verbose=True, n_epochs=1,
                                 trainable_layers=[False] * 12,
                                 trainable_old_embeddings=False,
                                 trainable_new_embeddings=True,
                                 val_size=0.0,
                                 val_interval=15000,)
    model.fit(dataset.Text1, dataset.Text2, Y=None)
    embeddings = model.sess.run(find_trainable_variables("we")[0])
    np.save("embeddings", embeddings)

    tf.reset_default_graph()

    model = Comparison(
        verbose=True,
        n_epochs=3,
        trainable_layers=[False] * 12,
        trainable_old_embeddings=False,
        trainable_new_embeddings=False,
        init_embeddings_from_file="embeddings.npy")
    trainX1, testX1, trainX2, testX2, trainY, testY = train_test_split(dataset.Text1, dataset.Text2, dataset.Target,
                                                                       test_size=0.3, random_state=42)
    model.fit(trainX1, trainX2, trainY)
    accuracy = np.mean(model.predict(testX1, testX2) == testY)
    class_balance = np.mean(testY)
    print('Test Accuracy: {:0.2f} for a {:0.2f} class balance'.format(accuracy, class_balance))
