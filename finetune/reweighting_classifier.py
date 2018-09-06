import tensorflow as tf
import numpy as np

from finetune.sample_reweight import learning_to_reweight
from finetune.classifier import Classifier


class ReweightingMixin:
    def finetune(self, goldX, goldY, *args, **kwargs):
        encoded_gold = self._text_to_ids(goldX)
        self.goldX, self.goldM, self.goldY = encoded_gold.token_ids, encoded_gold.mask, goldY

        return super(ReweightingMixin, self).finetune(*args, **kwargs)

    def model_body(self, X, M, Y, target_dim, train, do_reuse, compile_lm):
        encoded_gold_y = self.label_encoder.transform(self.goldY)
        num_batches = len(encoded_gold_y)//self.config.batch_size + (1 if len(encoded_gold_y)%self.config.batch_size != 0 else 0)

        g_split_x = np.array_split(self.goldX, num_batches)
        g_split_m = np.array_split(self.goldM, num_batches)
        g_split_y = np.array_split(encoded_gold_y, num_batches)
        self._i = 0

        def gold_input_fn():
            self._i = (self._i + 1) % num_batches
            i = self._i
            return np.int32(g_split_x[i]), np.float32(g_split_m[i]), np.float32(g_split_y[i])

        goldX, goldM, goldY = tf.py_func(gold_input_fn, [], [tf.int32, tf.float32, tf.float32])
        goldX.set_shape([None] + list(g_split_x[0].shape[1:]))
        goldM.set_shape([None] + list(g_split_m[0].shape[1:]))
        goldY.set_shape([None] + list(g_split_y[0].shape[1:]))

        feat_state, lm_tate, tm_state = super(ReweightingMixin, self).model_body(X, M, Y, target_dim, train, do_reuse,
                                                                                 compile_lm)

        # TODO add a tensorflow generator function to feed gold standard values.

        def model_fn(features, targets):
            _, _, target_model_state = super(ReweightingMixin, self).model_body(features, goldM, targets,
                                                                                target_dim, train, do_reuse,
                                                                                compile_lm)
            return target_model_state["logits"], target_model_state["losses"]

        weights = learning_to_reweight(goldX, goldY, X, Y, model_fn, lr=self.config["lr"])
        tm_state["losses"] *= weights
        return feat_state, lm_tate, tm_state

class ReweightingClassifier(ReweightingMixin, Classifier):
    pass
