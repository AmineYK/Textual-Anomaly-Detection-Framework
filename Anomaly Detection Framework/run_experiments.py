import os

# === Paramètres ===
list_dataset_name = ["20newsgroups", "reuters"]
# list_list_inlier_topic = [
#     ["computer"],
#     ["earn"]
# ]

list_list_inlier_topic = [
    ["computer", "recreation", "science", "miscellaneous", "politics", "religion"],
    ["earn", "acq", "crude", "trade", "money-fx", "interest", "ship"]
]

list_embeddings = ["glove", "fasttext"]
list_models = ["ocsvm"]

for i, dataset_name in enumerate(list_dataset_name):
    list_inlier_topic = list_list_inlier_topic[i]
    for inlier_topic in list_inlier_topic:
        for embedding in list_embeddings:
            for ad_model in list_models:

                cmd = (
                    f"python3 main_all_runs.py "
                    f"--dataset_name {dataset_name} "
                    f"--training_mode one_class "
                    f"--inlier_topic {inlier_topic} "
                    f"--type_tac ruff "
                    f"--anomaly_rate 0.1 "
                    f"--emb_model {embedding}_300d.kv "
                    f"--type_emb {embedding} "
                    f"--batch_size 64 "
                    f"--shuffle "
                    f"--ad_model {ad_model} "
                )

                if ad_model == "cvdd":
                    cmd += (
                        f"--attention_size 150 "
                        f"--n_attention_heads 10 "
                        f"--lambda_p 1.0 "
                        f"--alpha_scheduler logarithmic "
                        f"--n_epochs 20 "
                        f"--lr 0.01 "
                        f"--lr_milestones 10 15"
                    )
                elif ad_model == "ocsvm":
                    cmd += (
                        f"--nu 0.05 "
                        f"--kernel rbf "
                        f"--gamma 0.1"
                    )

                # # === Log & exécution ===
                # os.makedirs("logs", exist_ok=True)
                # log_file = f"logs/{dataset_name}_{inlier_topic}_{embedding}_{ad_model}.txt"

                # print(f"\n🚀 Running: {cmd}")
                # print(f"📝 Log file: {log_file}")

                # os.system(f"{cmd} > {log_file} 2>&1")
                print(f"\nRunning: {cmd}")
                os.system(cmd)

