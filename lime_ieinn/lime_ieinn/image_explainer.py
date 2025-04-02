import numpy as np
import copy
from sklearn.metrics import pairwise_distances
from .base import LocalExplainer
from .utility import exponential_kernel, segment_image_grid

class ImageExplainer:
    def __init__(self, kernel_width=0.25, additivity_order=1, epochs=100):
        self.kernel_fn = lambda d: exponential_kernel(d, kernel_width)
        self.explainer = LocalExplainer(self.kernel_fn, additivity_order, epochs)

    def explain_instance(self, image, predict_fn, num_samples=1000, num_features=5, grid_size=4):
        original_was_grayscale = len(image.shape) == 2

        image, segments = segment_image_grid(image, grid_size=grid_size)

        n_segments = np.unique(segments).shape[0]
        data = np.random.randint(0, 2, size=(num_samples, n_segments))
        data[0, :] = 1  # 元画像に対応

        perturbed_images = []
        for row in data:
            temp = copy.deepcopy(image)
            mask = np.zeros(segments.shape, dtype=bool)
            for i, on in enumerate(row):
                if on == 0:
                    mask[segments == i] = True
            temp[mask] = np.mean(image[~mask], axis=0) if np.any(~mask) else temp[mask]
            perturbed_images.append(temp)

        perturbed_images = np.array(perturbed_images)

        if original_was_grayscale:
            perturbed_images = perturbed_images.mean(axis=-1)

        labels = predict_fn(perturbed_images)
        top_label = int(np.argmax(labels[0, :]))
        distances = pairwise_distances(data, data[0].reshape(1, -1), metric='cosine').ravel()

        explanation = self.explainer.explain(
            data, labels, distances, label=top_label, num_features=num_features
        )

        explanation.segments = segments
        explanation.original_image = np.array(image)
        explanation.grid_size = grid_size

        return explanation
