import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

data = requests.get(
    "http://localhost:8000/latest_embeddings_json"
).json()

embeddings = np.array(data["embeddings"])
labels = np.array(data["actual_labels"])

print("Embedding shape:", embeddings.shape)

# Convert category names to numbers for coloring
unique_labels = np.unique(labels)

label_to_number = {
    label: i for i, label in enumerate(unique_labels)
}

numeric_labels = np.array(
    [label_to_number[label] for label in labels]
)

print("Number of categories:", len(unique_labels))
print("Categories:", unique_labels)

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    embeddings[:, 0],
    embeddings[:, 1],
    c=numeric_labels,
    cmap="tab10",
    s=8,
    alpha=0.7
)

# Add category legend
legend_elements = [
    mpatches.Patch(
        color=plt.cm.tab10(i),
        label=label
    )
    for i, label in enumerate(unique_labels)
]

plt.legend(
    handles=legend_elements,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    title="Categories"
)

plt.title(
    f"Representation Space - Iteration {data['iteration']}"
)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.tight_layout()

plt.savefig(
    "representation_space_by_category.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
