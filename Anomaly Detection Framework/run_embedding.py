import os
import shutil
import sys
sys.path.append('./Textual-Anomaly-Detection-Framework/Anomaly Detection Framework')

from Data_Preparation.Embedding import embedding_encoder
from utils import load_data_inlier, load_data_test

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SAVE_DIR = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data"
MODEL_NAME = "sentence-transformers/sentence-t5-large"
EMBEDDING_COL = "st5_large_embedding"
NB_RUNS_DEFAULT = 5

NB_RUNS = {
    '20newsgroups': 5,
    'reuters':      5,
    'agnews':       5,
    'dbpedia14':    4,
    'sms':          5,
    'enron':        5,
    'imdb':         4,
    'sst2':         4,
    'mage':         5,
}

DATASETS = {
    '20newsgroups': ['computer', 'recreation', 'science', 'miscellaneous', 'politics', 'religion'],
    'reuters':      ['earn', 'trade', 'acq', 'money-fx', 'crude', 'ship', 'interest'],
    'agnews':       ['World', 'Sports', 'Business', 'Sci-Tech'],
    'sms':          ['normal'],
    'enron':        ['normal'],
    'imdb':         ['positive', 'negative'],
    'sst2':         ['positive', 'negative'],
    'dbpedia14':    ["Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
                     "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
                     "Plant", "Album", "Film", "Written Work"],
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def get_encoder(device):
    return embedding_encoder.EmbeddingEncoder(
        MODEL_NAME,
        EMBEDDING_COL,
        'sentencebert',
        device
    )


def safe_save(dataset, path):
    tmp_path = path + "_tmp"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    dataset.save_to_disk(tmp_path)
    if os.path.exists(path):
        shutil.rmtree(path)
    shutil.move(tmp_path, path)


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

def encode_train(device):
    print("\n" + "="*60)
    print("ENCODING TRAIN DATASETS")
    print("="*60)

    for dataset_name, inlier_topics in DATASETS.items():
        print(f"\n📂 Dataset : {dataset_name}")

        for inlier_topic in inlier_topics:
            print(f"  ▶ {inlier_topic}...")

            path = os.path.join(
                SAVE_DIR,
                f"{dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_cvdd.pt"
            )

            dataset = load_data_inlier(dataset_name, inlier_topic, SAVE_DIR, is_infec=False, is_cvdd=True)

            if EMBEDDING_COL in dataset.column_names:
                print("    ⏭ Embedding already exists, skipping...")
                continue

            encoder = get_encoder(device)
            dataset = encoder.forward(dataset, 'content')

            safe_save(dataset, path)
            print(f"    ✅ Saved: {path}")


# ─────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────

def encode_test(device):
    print("\n" + "="*60)
    print("ENCODING TEST DATASETS")
    print("="*60)

    for dataset_name, inlier_topics in DATASETS.items():
        print(f"\n📂 Dataset : {dataset_name}")

        for inlier_topic in inlier_topics:
            print(f"\n  ==== {inlier_topic} ====")

            for run in range(NB_RUNS[dataset_name]):
                print(f"  Run {run+1}...")

                path = os.path.join(
                    SAVE_DIR,
                    f"{dataset_name}/{inlier_topic}/run{run+1}",
                    f"ds_test_{inlier_topic}_cvdd_run{run+1}.pt"
                )

                dataset = load_data_test(dataset_name, inlier_topic, run+1, SAVE_DIR, is_cvdd=True)

                if EMBEDDING_COL in dataset.column_names:
                    print("    ⏭ Embedding already exists, skipping...")
                    continue

                encoder = get_encoder(device)
                dataset = encoder.forward(dataset, 'content')

                safe_save(dataset, path)
                print(f"    ✅ Saved: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import torch
    import argparse

    parser = argparse.ArgumentParser(description="Encode datasets with ST5-large")
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "both"],
        default="both",
        help="Which split to encode (default: both)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥  Device : {device}")
    print(f"🤖 Model  : {MODEL_NAME}")
    print(f"📌 Column : {EMBEDDING_COL}")

    if args.split in ("train", "both"):
        encode_train(device)

    if args.split in ("test", "both"):
        encode_test(device)

    print("\n🎉 Done !")