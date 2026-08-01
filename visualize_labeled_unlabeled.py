import requests
import numpy as np
import matplotlib.pyplot as plt

data = requests.get(
    "http://localhost:8000/latest_embeddings_json"
).json()

embeddings = np.array(data["embeddings"])
is_labeled = np.array(data["is_labeled"])

print("Embedding shape:", embeddings.shape)
print("Labeled samples:", np.sum(is_labeled))
print("Unlabeled samples:", np.sum(~is_labeled))

plt.figure(figsize=(10, 7))

# Unlabeled samples
plt.scatter(
    embeddings[~is_labeled, 0],
    embeddings[~is_labeled, 1],
    s=8,
    alpha=0.4,
    label="Unlabeled"
)

# Labeled samples
plt.scatter(
    embeddings[is_labeled, 0],
    embeddings[is_labeled, 1],
    s=15,
    alpha=0.8,
    label="Labeled"
)

plt.title(
    f"Labeled vs Unlabeled Representation Space - Iteration {data['iteration']}"
)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.legend()

plt.tight_layout()

plt.savefig(
    "representation_space_labeled_unlabeled.png",
    dpi=300
)

plt.show()
