import os

# === Paramètres ===
list_dataset_name = ["reuters"]
# list_dataset_name = ["20newsgroups", "reuters"]

list_list_inlier_topic = [
    # ["computer", "recreation", "science", "miscellaneous", "politics", "religion"],
    ["earn", "acq", "crude", "trade", "money-fx", "interest", "ship"]
]

list_embeddings = ["glove"]
# list_embeddings = ["glove", "fasttext"]
list_models = ["cvdd"]

# ocsvm_params_dict = {
#     "20newsgroups": {
#         "computer": {"nu": 0.05, "kernel": "rbf", "gamma": 0.5},
#         "recreation": {"nu": 0.1, "kernel": "rbf", "gamma": 1.0},
#         "science": {"nu": 0.05, "kernel": "rbf", "gamma": 0.8},
#         "miscellaneous": {"nu": 0.1, "kernel": "linear", "gamma": 1.0},
#         "politics": {"nu": 0.05, "kernel": "rbf", "gamma": 0.6},
#         "religion": {"nu": 0.05, "kernel": "linear", "gamma": 1.0},
#     },
#     "reuters": {
#         "earn": {"nu": 0.05, "kernel": "rbf", "gamma": 1.0},
#         "acq": {"nu": 0.1, "kernel": "rbf", "gamma": 0.8},
#         "crude": {"nu": 0.05, "kernel": "linear", "gamma": 1.0},
#         "trade": {"nu": 0.05, "kernel": "rbf", "gamma": 0.5},
#         "money-fx": {"nu": 0.05, "kernel": "rbf", "gamma": 0.6},
#         "interest": {"nu": 0.1, "kernel": "linear", "gamma": 1.0},
#         "ship": {"nu": 0.05, "kernel": "rbf", "gamma": 0.7},
#     }
# }

ocsvm_params_dict = {
    '20newsgroups': {
        'computer':      {'kernel': 'sigmoid', 'nu': 0.1,  'gamma': 1},
        'recreation':    {'kernel': 'sigmoid', 'nu': 0.1,  'gamma': 1},
        'science':       {'kernel': 'rbf',     'nu': 0.1,  'gamma': 0.001},
        'miscellaneous': {'kernel': 'sigmoid', 'nu': 0.05, 'gamma': 1},
        'politics':      {'kernel': 'rbf',     'nu': 0.15, 'gamma': 1},
        'religion':      {'kernel': 'sigmoid', 'nu': 0.05, 'gamma': 1}
    },
    'reuters': {
        'earn':     {'kernel': 'sigmoid', 'nu': 0.05, 'gamma': 0.001},
        'acq':      {'kernel': 'rbf',     'nu': 0.05, 'gamma': 1},
        'crude':    {'kernel': 'rbf',     'nu': 0.15, 'gamma': 1},
        'trade':    {'kernel': 'rbf',     'nu': 0.05, 'gamma': 1},
        'money-fx': {'kernel': 'rbf',     'nu': 0.05, 'gamma': 1},
        'interest': {'kernel': 'rbf',     'nu': 0.05, 'gamma': 0.1},
        'ship':     {'kernel': 'sigmoid', 'nu': 0.1,  'gamma': 0.01}
    }
}


for i, dataset_name in enumerate(list_dataset_name):
    list_inlier_topic = list_list_inlier_topic[i]
    for inlier_topic in list_inlier_topic:
        for embedding in list_embeddings:
            for ad_model in list_models:

                cmd = (
                    f"python3 main_all_runs.py "
                    f"--dataset_name {dataset_name} "
                    f"--training_mode one_class "
                    f"--device cuda "
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
                        f"--n_epochs 40 "
                        f"--lr 0.01 "
                        f"--lr_milestones 20 30"
                    )
                elif ad_model == "ocsvm":
                    params = ocsvm_params_dict[dataset_name][inlier_topic]
                    cmd += f"--nu {params['nu']} --kernel {params['kernel']} --gamma {params['gamma']}"


                # # === Log & exécution ===
                # os.makedirs("logs", exist_ok=True)
                # log_file = f"logs/{dataset_name}_{inlier_topic}_{embedding}_{ad_model}.txt"

                # print(f"\n🚀 Running: {cmd}")
                # print(f"📝 Log file: {log_file}")

                # os.system(f"{cmd} > {log_file} 2>&1")
                print(f"\nRunning: {cmd}")
                os.system(cmd)

