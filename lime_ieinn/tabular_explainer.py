import numpy as np
from sklearn.metrics import pairwise_distances
from .base import LocalExplainer
from .utility import exponential_kernel

class TabularExplainer:
    def __init__(self, kernel_width=1.0, additivity_order=1, epochs=100, feature_names=None):
        self.kernel_fn = lambda d: exponential_kernel(d, kernel_width)
        self.explainer = LocalExplainer(self.kernel_fn, additivity_order, epochs)
        self.feature_names = feature_names

    def explain_instance(self, instance, predict_fn, reference_data,
                         num_samples=500, num_features=10):
        n_features = instance.shape[0]
        means = np.mean(reference_data, axis=0)

        binary_mask = np.random.randint(0, 2, size=(num_samples, n_features))
        binary_mask[0] = 1

        data = binary_mask * instance + (1 - binary_mask) * means
        labels = predict_fn(data)
        if labels.ndim == 1:  # 出力が1次元（例: 回帰や2値分類の確率なしクラスラベル）
            labels = labels.reshape(-1, 1)
            top_label = 0
        else:
            top_label = int(np.argmax(labels[0, :]))
        distances = pairwise_distances(binary_mask, binary_mask[0].reshape(1, -1), metric='euclidean').ravel()

        explanation = self.explainer.explain(
            binary_mask, labels, distances, label=top_label, num_features=num_features
        )
        explanation.feature_names = self.feature_names

        return explanation