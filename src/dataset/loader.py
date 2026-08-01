import librosa, hashlib, os, json
import librosa.display
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from config import INITIAL_LABELS_PER_CLASS_COUNT, DATA_DIR, SAMPLE_RATE, TARGET_DURATION, EMBEDDING_DIM, MAX_PER_CLASS_PER_FOLD
from dataset.cache_dataset import UrbanSoundCachedEmbeddingDataset
from utils.logging_utils import get_logger, log_duration
log = get_logger("loader")

def _hash_paths(paths):
    h = hashlib.sha1()
    for p in paths:
        h.update(str(p).encode())
    return h.hexdigest()[:8]

# PyTorch dataset
class UrbanSoundEmbeddingDataset(Dataset):
    def __init__(self, paths, labels,original_labels, embedder):
        self.paths = paths
        self.labels = labels
        self.original_labels = original_labels
        self.embedder = embedder

    def __getitem__(self, idx):
        audio = self.load_audio_file(self.paths[idx])
        embedding = self.embedder.get_embedding(audio)
        filename = os.path.basename(self.paths[idx])
        return (
            torch.tensor(embedding, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long),
            self.original_labels[idx],
            filename,
            idx
        )

    def __len__(self):
        return len(self.paths)

    def load_audio_file(self, path, sr=SAMPLE_RATE, duration=TARGET_DURATION):
        waveform, _ = librosa.load(path, sr=sr, mono=True)
        target_length = int(sr * duration)
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))

        return waveform

class UrbanSoundLoader:
    def __init__(self, embedder):
        self.embedder = embedder
        self.metadata_path = os.path.join(DATA_DIR, "metadata", "UrbanSound8K.csv")
        self.cache_dir = Path(DATA_DIR) / "embeddings_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_or_load_embeddings(self, paths, split_key):
        # Build cache file names
        key = f"{split_key}_{_hash_paths(paths)}_sr{SAMPLE_RATE}_dur{TARGET_DURATION}"
        emb_path = self.cache_dir / f"{key}.npy"
        meta_path = self.cache_dir / f"{key}.json"

        # If cache exists and matches config, load it
        if emb_path.exists() and meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if (meta.get("sr") == SAMPLE_RATE and
                    float(meta.get("duration")) == float(TARGET_DURATION) and
                    meta.get("dim") == EMBEDDING_DIM and
                    meta.get("embedder") == "yamnet/1"):
                # memory-map for low RAM
                return np.load(emb_path, mmap_mode="r")

        # Otherwise compute and save
        log.info(f"Cache miss | {key} | computing embeddings n={len(paths)}")
        with log_duration(log, "compute_embeddings", split=split_key, n=len(paths)):

            E = np.empty((len(paths), EMBEDDING_DIM), dtype=np.float32)
            for i, p in enumerate(paths):
                y, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
                target_len = int(SAMPLE_RATE * TARGET_DURATION)
                if len(y) > target_len:
                    y = y[:target_len]
                else:
                    y = np.pad(y, (0, target_len - len(y)))
                E[i] = self.embedder.get_embedding(y)  # -> (1024,)

        np.save(emb_path, E)
        with open(meta_path, "w") as f:
            json.dump({
                "sr": SAMPLE_RATE,
                "duration": float(TARGET_DURATION),
                "dim": EMBEDDING_DIM,
                "embedder": "yamnet/1",
                "n": len(paths)
            }, f)
        log.info(f"Cache saved | {key}")
        return np.load(emb_path, mmap_mode="r")

    def _cap_per_class_per_fold(self, df, max_per_class_per_fold):
        if max_per_class_per_fold is None:
            return df

        log.info(f"Capping data | max_per_class_per_fold={max_per_class_per_fold}")

        return (
            df.groupby(["fold", "class_code"], group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), max_per_class_per_fold),
                    random_state=42
                )
            )
            .reset_index(drop=True)
        )

    def get_labeled_unlabeled_datasets(self, held_out_fold):
        df = pd.read_csv(self.metadata_path)
        df['class_code'] = df['classID'].astype(int)
        class_code_to_label = (
            df[['classID', 'class']]
            .drop_duplicates()
            .sort_values('classID')
            .set_index('classID')['class']
            .to_dict()
        )

        train_df = df[df['fold'] != held_out_fold]
        test_df = df[df['fold'] == held_out_fold]

        if MAX_PER_CLASS_PER_FOLD is not None:
            log.info(
                f"Capping training data | max_per_class_total={MAX_PER_CLASS_PER_FOLD}"
            )
            train_df = (
                train_df
                .groupby(["fold", "class_code"], group_keys=False)
                .apply(
                    lambda x: x.sample(
                        n=min(len(x), MAX_PER_CLASS_PER_FOLD),
                        random_state=42
                    )
                )
                .reset_index(drop=True)
            )

        log.info(f"DEBUG | train_df size after cap = {len(train_df)}")

        # Taking PER_CLASS_COUNT samples from every class for training
        labeled_df = (
            train_df
            .groupby("class_code", group_keys=False)
            .apply(
                lambda x: x.sample(
                    n=min(len(x), INITIAL_LABELS_PER_CLASS_COUNT),
                    random_state=42
                )
            )
        )
        unlabeled_df = train_df.drop(labeled_df.index)
        combined_df = pd.concat([labeled_df, unlabeled_df]).reset_index(drop=True)
        labeled_indices = list(range(len(labeled_df)))
        unlabeled_indices = list(range(len(labeled_df), len(combined_df)))

        train_paths = [Path(DATA_DIR) / "audio" / f"fold{row['fold']}" / row['slice_file_name'] for _, row in combined_df.iterrows()]
        train_labels = combined_df['class_code'].to_numpy()
        train_original_labels = combined_df['class'].tolist()
        train_filenames = [os.path.basename(p) for p in train_paths]
        #full_train_dataset = UrbanSoundEmbeddingDataset(train_paths, train_labels, train_original_labels, self.embedder)

        test_paths = [Path(DATA_DIR) / "audio" / f"fold{row['fold']}" / row['slice_file_name'] for _, row in test_df.iterrows()]
        test_labels = test_df['class_code'].to_numpy()
        test_original_labels = test_df['class'].tolist()
        test_filenames = [os.path.basename(p) for p in test_paths]

        # compute or load cached embeddings once
        train_embs = self._compute_or_load_embeddings(train_paths, split_key=f"train_ex_fold{held_out_fold}")
        test_embs = self._compute_or_load_embeddings(test_paths, split_key=f"test_fold{held_out_fold}")

        full_train_dataset = UrbanSoundCachedEmbeddingDataset(train_embs, train_labels, train_original_labels,
                                                              train_filenames)
        test_dataset = UrbanSoundCachedEmbeddingDataset(test_embs, test_labels, test_original_labels, test_filenames)

        return full_train_dataset, labeled_indices, unlabeled_indices, test_dataset, class_code_to_label


