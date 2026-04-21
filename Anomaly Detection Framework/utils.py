import re
from collections import defaultdict
import numpy as np
import os
import torch
from datetime import datetime
import datasets
from Modelisation.Baselines.CVDD.networks import model_bert as md
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

BASE_DIR = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results"
# =========================
# CONFIGURATION
# =========================

BASE_RESULTS_DIR = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results"
OUTPUT_LATEX_DIR = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results/latex_tables"

DATASETS = ["reuters","20newsgroups", "agnews", "dbpedia14", "sms", "enron", "sst2", "imdb", "m4"]
ENCODING_TYPES = ["sentence-bert", "bert", "fasttext"]
NU_VALUES = [0.0, 0.1]

os.makedirs(OUTPUT_LATEX_DIR, exist_ok=True)

# =========================
# MODELS & GROUPS
# =========================


MODEL_GROUPS = [
    ("Classical baselines", ["ocsvm", "AE"]),
    ("Deep baselines", ["RSRAE", "CVDD", "DATE", "FATE"]),
    ("Flow-based models", [
        "TCCM",
        "flow-matching",
        "flow-matching-Transformers",
        "flow-matching-Transformers-PP",
        "flow-matching-Transformers-Comp"
    ]),
]

# MODEL_LATEX = {
#     "ocsvm": "OCSVM",
#     "AE": "AE",
#     "RSRAE": "RSRAE",
#     "CVDD": "CVDD",
#     "DATE": "DATE",
#     "FATE": "FATE",
#     "TCCM": "TCCM",
#     "flow-matching": "\\textbf{BasicFM}",
#     "flow-matching-Transformers": "\\textbf{TranFM}",
#     "flow-matching-Transformers-PP": "\\textbf{TranFM-PP}"
# }

MODEL_LATEX_BY_ENCODING = {
    "sentence-bert": {
        "ocsvm": "OCSVM",
        "AE": "AE",
        "RSRAE": "RSRAE",
        "CVDD": "CVDD",
        "DATE": "DATE",
        "FATE": "FATE",
        "TCCM": "TCCM",
        "flow-matching": "\\textbf{BasicFM}",
        "flow-matching-Transformers": "\\textbf{TranFM}",
        "flow-matching-Transformers-PP": "\\textbf{TranFM-PP}",
    },

    "bert": {
        "ocsvm": "OCSVM",
        "AE": "AE",
        "RSRAE": "RSRAE",
        "CVDD": "CVDD",
        "TCCM": "TCCM",
        "flow-matching-Transformers-Comp": "FMTToken-Sentence",
    }
}

MODEL_ORDER = [m for _, g in MODEL_GROUPS for m in g]

# =========================
# REGEX PATTERNS
# =========================

pattern = {
    "dataset": re.compile(r"Dataset:\s*(.*)"),
    "inlier": re.compile(r"Inlier class:\s*(.*)"),
    "model": re.compile(r"AD model:\s*(.*)"),
    "type_emb": re.compile(r"Embedding type:\s*(.*)"),
    "auc": re.compile(r"AUC:\s*([\d.]+)\s*±\s*([\d.]+)"),
    "ap": re.compile(r"Avg Precision:\s*([\d.]+)\s*±\s*([\d.]+)"),
    "fpr95": re.compile(r"FPR@95:\s*([\d.]+)\s*±\s*([\d.]+)"),
}

# =========================
# UTILS
# =========================

def fmt(mean, std):
    return f"{mean:.4f} $\\pm$ {std:.4f}"

def get_best_two(values, maximize=True):
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 2:
        return None, None

    valid = sorted(valid, key=lambda x: x[1], reverse=maximize)
    return valid[0][0], valid[1][0]

def iter_models_with_separators():
    for i, (_, group) in enumerate(MODEL_GROUPS):
        for model in group:
            yield model
        if i < len(MODEL_GROUPS) - 1:
            yield "__SEP__"

# =========================
# LOAD RESULTS
# =========================
def load_results_for_config(encoding, nu, metric="auc"):
    """
    results[type_emb][dataset][inlier][model] = (mean, std)
    """
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    base_dir = os.path.join(BASE_RESULTS_DIR, encoding, f"nu_{nu}")

    for ds in DATASETS:
        path = os.path.join(base_dir, f"{ds}.txt")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            block = dict.fromkeys(["type_emb", "dataset", "inlier", "model", "metric"])

            for line in f:
                line = line.strip()

                if m := pattern["type_emb"].search(line):
                    block["type_emb"] = m.group(1)

                if m := pattern["dataset"].search(line):
                    block["dataset"] = m.group(1)

                if m := pattern["inlier"].search(line):
                    block["inlier"] = m.group(1)

                if m := pattern["model"].search(line):
                    block["model"] = m.group(1)

                if m := pattern[metric].search(line):
                    block["metric"] = (float(m.group(1)), float(m.group(2)))

                if line.startswith("===="):
                    if all(block.values()):
                        results[
                            block["type_emb"]
                        ][
                            block["dataset"]
                        ][
                            block["inlier"]
                        ][
                            block["model"]
                        ] = block["metric"]

                    block = dict.fromkeys(block)

    return results
# =========================
# TABLE GENERATION
# =========================

def generate_global_table(results, encoding, metric_name="AUC"):
    datasets = list(results.keys())

    global_means = {m: [] for m in MODEL_ORDER}
    global_stds = {m: [] for m in MODEL_ORDER}

    for model in MODEL_ORDER:
        for ds in datasets:
            vals = [
                results[ds][ic][model]
                for ic in results[ds]
                if model in results[ds][ic]
            ]
            if vals:
                global_means[model].append(np.mean([v[0] for v in vals]))
                global_stds[model].append(np.mean([v[1] for v in vals]))
            else:
                global_means[model].append(None)
                global_stds[model].append(None)

    best_idx, sec_idx = {}, {}
    for c in range(len(datasets)):
        col_vals = [global_means[m][c] for m in MODEL_ORDER]
        maximize = metric_name.lower() != "fpr95"
        b, s = get_best_two(col_vals, maximize=maximize)
        best_idx[c], sec_idx[c] = b, s

    latex = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{metric_name} on all datasets}}"
        "\\resizebox{1.1\\textwidth}{!}{",
        "\\begin{tabular}{l" + "c" * len(datasets) + "}",
        "\\toprule",
        "Model & " + " & ".join(ds.capitalize() for ds in datasets) + " \\\\",
        "\\midrule",
    ]

    row_id = 0
    for item in iter_models_with_separators():
        if item == "__SEP__":
            latex.append("\\midrule")
            continue

        # row = [MODEL_LATEX[item]]
        row = [MODEL_LATEX_BY_ENCODING[encoding].get(item, item)]
        for c in range(len(datasets)):
            mean = global_means[item][c]
            std = global_stds[item][c]

            if mean is None:
                cell = "—"
            else:
                cell = fmt(mean, std)

                if MODEL_ORDER.index(item) == best_idx[c]:
                    cell = f"\\textbf{{{cell}}}"
                elif MODEL_ORDER.index(item) == sec_idx[c]:
                    cell = f"\\underline{{{cell}}}"

                # if item == "flow-matching":
                #     cell = f"\\textbf{{{cell}}}"

            row.append(cell)

        latex.append(" & ".join(row) + " \\\\")
        row_id += 1

    latex.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\label{tab:global}",
        "\\end{table}",
    ])

    return "\n".join(latex)

def generate_dataset_tables(results, encoding, metric_name="AUC"):
    all_tables = []

    for ds in results:
        inliers = list(results[ds].keys())

        auc_matrix = {
            m: [
                results[ds][ic].get(m, (None, None))[0]
                for ic in inliers
            ]
            for m in MODEL_ORDER
        }

        best_idx, sec_idx = {}, {}
        for c in range(len(inliers)):
            col_vals = [auc_matrix[m][c] for m in MODEL_ORDER]
            maximize = metric_name.lower() != "fpr95"
            b, s = get_best_two(col_vals, maximize=maximize)
            best_idx[c], sec_idx[c] = b, s

        latex = [
            "\\begin{table}[H]",
            "\\centering",
            f"\\caption{{{metric_name} on {ds.capitalize()}}}"
            "\\resizebox{1.1\\textwidth}{!}{",
            "\\begin{tabular}{l" + "c" * len(inliers) + "}",
            "\\toprule",
            "Model & " + " & ".join(inliers) + " \\\\",
            "\\midrule",
        ]

        for item in iter_models_with_separators():
            if item == "__SEP__":
                latex.append("\\midrule")
                continue

            # row = [MODEL_LATEX[item]]
            row = [MODEL_LATEX_BY_ENCODING[encoding].get(item, item)]
            for c, ic in enumerate(inliers):
                if item in results[ds][ic]:
                    mean, std = results[ds][ic][item]
                    cell = fmt(mean, std)

                    if MODEL_ORDER.index(item) == best_idx[c]:
                        cell = f"\\textbf{{{cell}}}"
                    elif MODEL_ORDER.index(item) == sec_idx[c]:
                        cell = f"\\underline{{{cell}}}"

                    # if item == "flow-matching":
                    #     cell = f"\\textbf{{{cell}}}"
                else:
                    cell = "—"

                row.append(cell)

            latex.append(" & ".join(row) + " \\\\")

        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            f"\\label{{tab:{ds}}}",
            "\\end{table}",
        ])

        all_tables.append("\n".join(latex))

    return "\n\n".join(all_tables)

# =========================
# MAIN ENTRY
# =========================

def generate_tables_for_config(encoding, nu, metric="auc"):
    results = load_results_for_config(encoding, nu, metric=metric)
    out_tex = os.path.join(
        OUTPUT_LATEX_DIR,
        f"tables_{metric}_{encoding.replace('-', '')}_nu_{nu}.tex"
    )

    with open(out_tex, "w") as f:
        for type_emb in results:
            title = type_emb.replace("_", " ").title()
            f.write(f"\\subsection {{{title}}}\n\n")
            f.write(generate_global_table(results[type_emb], encoding, metric_name=metric.upper()))
            f.write("\n\n")
            f.write(generate_dataset_tables(results[type_emb], encoding, metric_name=metric.upper()))
            f.write("\n\n")

    print(f"[OK] Generated {out_tex}")

def save_results(
    dataset_name,
    inlier_topic,
    type_emb,
    ad_model,
    auc_mean, ap_mean, fpr_mean,
    auc_std=None, ap_std=None, fpr_std=None,
    train_time=None,
    nu=0.0,
    output_dir=BASE_DIR,
    overwrite='smart'
):  
    # --- Création du dossier spécifique : <output_dir>/<embedding>/nu_<value>/ ---
    dataset_folder = os.path.join(output_dir, type_emb, f"nu_{nu}")
    os.makedirs(dataset_folder, exist_ok=True)

    # --- Nom du fichier : <dataset>.txt ---
    filepath = os.path.join(dataset_folder, f"{dataset_name}.txt")

    # --- Lire le contenu existant ---
    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            existing_content = f.read()

    # --- Pattern pour identifier un bloc existant ---
    pattern = (
        rf"Dataset:\s*{re.escape(dataset_name)}\s*"
        rf"Inlier class:\s*{re.escape(inlier_topic)}\s*"
        rf"Embedding type:\s*{re.escape(type_emb)}\s*"
        rf"AD model:\s*{re.escape(ad_model)}"
    )

    # --- Formatage "mean ± std" ---
    def fmt(mean, std):
        return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"

    time_str = f"{train_time:.2f} sec" if train_time is not None else "N/A"

    # --- Nouveau bloc à écrire ---
    new_block = (
        "========================================\n"
        f"Dataset:        {dataset_name}\n"
        f"Inlier class:   {inlier_topic}\n"
        f"Embedding type: {type_emb}\n"
        f"AD model:       {ad_model}\n"
        f"Training time:  {time_str}\n"
        "----------------------------------------\n"
        f"AUC:            {fmt(auc_mean, auc_std)}\n"
        f"Avg Precision:  {fmt(ap_mean, ap_std)}\n"
        f"FPR@95:         {fmt(fpr_mean, fpr_std)}\n"
        "========================================\n\n"
    )

    # --- Vérification si le bloc existe déjà ---
    match = re.search(pattern, existing_content)
    if match:
        old_block_pattern = (
            r"========================================\n"
            + pattern +
            r".*?========================================\n\n"
        )
        old_block = re.search(old_block_pattern, existing_content, flags=re.DOTALL)

        old_auc = -1
        if old_block:
            old_block_text = old_block.group(0)
            old_auc_match = re.search(r"AUC:\s*([\d.]+)", old_block_text)
            old_auc = float(old_auc_match.group(1)) if old_auc_match else -1

        do_replace = False
        if overwrite == "naive":
            do_replace = True
        elif overwrite == "smart" and auc_mean > old_auc:
            do_replace = True
        elif overwrite is None:
            do_replace = False

        if do_replace:
            existing_content = re.sub(
                old_block_pattern, new_block, existing_content, flags=re.DOTALL
            )
            print(f"Résultats mis à jour pour ({dataset_name}, {inlier_topic}, {type_emb}, {ad_model}, nu={nu}).")
        else:
            print(f"Résultats existants non modifiés pour ({dataset_name}, {inlier_topic}, {type_emb}, {ad_model}, nu={nu}).")
            return
    else:
        existing_content += new_block
        print(f"Nouveaux résultats ajoutés pour ({dataset_name}, {inlier_topic}, {type_emb}, {ad_model}, nu={nu}).")

    # --- Écriture finale ---
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(existing_content)



def save_hyperparameters(dataset_name, inlier_topic,
                         batch_size=32,
                         latent_dim=256,
                         sinu=False,
                         batchnorm=False,
                         dropout=0.1,
                         lr=1e-4,
                         weight_decay=1e-6,
                         n_epochs=25,
                         target=None,
                         source='sphere',
                         save_dir="Results"):
    
    os.makedirs(save_dir, exist_ok=True)

    filename = "hyperparams.txt"
    filepath = os.path.join(save_dir, filename)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    content = [
        "========================================",
        f"Run date : {now}",
        f"dataset_name : {dataset_name}",
        f"inlier_topic : {inlier_topic}",
        "",
        "Hyperparameters :",
        f"batch_size : {batch_size}",
        f"latent_dim : {latent_dim}",
        f"sinu : {sinu}",
        f"batchnorm : {batchnorm}",
        f"dropout : {dropout}",
        f"lr : {lr}",
        f"weight_decay : {weight_decay}",
        f"n_epochs : {n_epochs}",
        f"source : {source}",
        f"target : {target}",
        "========================================",
        "\n"
    ]

    with open(filepath, "a") as f:
        f.write("\n".join(content))

    print(f"Hyperparameters saved to: {filepath}")
    
    
    
import re

def load_hyperparams(dataset_name, inlier_topic, type_emb, file_path):
    """
    Lit un fichier de logs contenant plusieurs runs et renvoie les hyperparamètres
    correspondant au dataset et inlier_topic donnés.
    """
    with open(file_path, "r") as f:
        content = f.read()

    # On sépare chaque run via les blocs "====="
    blocks = [b.strip() for b in content.split("========================================") if b.strip()]

    # Regex pour matcher dataset + inlier_topic
    ds_pattern = re.compile(r"dataset_name\s*:\s*(.*)")
    inlier_pattern = re.compile(r"inlier_topic\s*:\s*(.*)")
    typeemb_pattern = re.compile(r"type_emb\s*:\s*(.*)")

    # Hyperparams regex
    hyper_pattern = re.compile(r"(\w+)\s*:\s*(.*)")

    for block in blocks:
        # Vérifier dataset_name et inlier_topic
        ds = ds_pattern.search(block)
        it = inlier_pattern.search(block)
        te = typeemb_pattern.search(block)


        if not ds or not it:
            continue

        if ds.group(1).strip() == dataset_name and it.group(1).strip() == inlier_topic and te.group(1).strip() == type_emb:
            # Extraire les hyperparamètres dans la section "Hyperparameters :"
            hypers = {}
            in_hyper_section = False

            for line in block.splitlines():
                line = line.strip()

                if line.startswith("Hyperparameters"):
                    in_hyper_section = True
                    continue

                if in_hyper_section:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()

                        # Convertir automatiquement en type Python
                        if value.isdigit():
                            value = int(value)
                        else:
                            try:
                                value = float(value)
                            except:
                                if value.lower() == "true":
                                    value = True
                                elif value.lower() == "false":
                                    value = False
                                else:
                                    value = value  # string brut

                        hypers[key] = value

            return hypers

    return None  # Aucun match


def load_data_inlier(dataset_name, inlier_topic, save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data", is_infec=False, is_cvdd=False):

    if is_cvdd:
        path = os.path.join(save_dir, f"{dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_cvdd.pt")
    else:
        if is_infec:
            path = os.path.join(save_dir, f"{dataset_name}/{inlier_topic}/ds_train_{inlier_topic}_infec.pt")
        else:
            path = os.path.join(save_dir, f"{dataset_name}/{inlier_topic}/ds_train_{inlier_topic}.pt")

    if not is_cvdd : return torch.load(path)['X_inlier']
    else: return datasets.load_from_disk(path)

def load_data_test(dataset_name, inlier_topic, n_run, save_dir = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Data",is_cvdd=False):

    if is_cvdd:
        path = os.path.join(save_dir, f"{dataset_name}/{inlier_topic}/run{n_run}/ds_test_{inlier_topic}_cvdd_run{n_run}.pt")
        return datasets.load_from_disk(path)
    else:
        path = os.path.join(save_dir, f"{dataset_name}/{inlier_topic}/run{n_run}/ds_test_{inlier_topic}_run{n_run}.pt")
        ds = torch.load(path)      
        return ds['X_test'], ds['y_test'] 
    

def load_fasttext_vec(path):
    embeddings = {}
    with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        first_line = f.readline().split()
        if len(first_line) == 2:
            dim = int(first_line[1])
        else:
            dim = len(first_line) - 1
            embeddings[first_line[0]] = np.array(first_line[1:], dtype=np.float32)

        for line in f:
            values = line.rstrip().split(' ')
            word = values[0]
            vec = np.asarray(values[1:], dtype=np.float32)
            embeddings[word] = vec

    return embeddings, dim

def encode_text(text, ft_vectors, dim):
    words = text.lower().split()
    vectors = [ft_vectors[w] for w in words if w in ft_vectors]

    if len(vectors) == 0:
        return np.zeros(dim)

    return np.mean(vectors, axis=0)


def get_data_fasttext(data, ft_path, device):
    ft_vectors, emb_dim = load_fasttext_vec(ft_path)

    X = np.vstack([
        encode_text(text, ft_vectors, emb_dim)
        for text in data['text']
    ])

    return torch.tensor(X, dtype=torch.float32).to(device)


def get_data_bert(bertname, data, device, is_train=False):

    tokenizer = AutoTokenizer.from_pretrained(bertname)
    bert = AutoModel.from_pretrained(bertname)

    dataset = md.CVDDDataset(
        data['text'],
        data['anomaly_class'],
        tokenizer
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=is_train)

    bert = bert.to(device)
    bert.eval()

    X = []
    y = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if not is_train: labels = batch["label"]

            outputs = bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            embeddings = outputs.last_hidden_state[:, 0, :] 

            X.append(embeddings.cpu())
            if not is_train:  y.append(labels.cpu())

    X = torch.cat(X, dim=0) 
    if not is_train: y = torch.cat(y, dim=0)

    if not is_train:
        return X, y
    else:
        return X
      




