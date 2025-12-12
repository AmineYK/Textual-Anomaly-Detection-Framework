import re
from collections import defaultdict
import numpy as np
import os
import torch
from datetime import datetime
import datasets


BASE_DIR = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results"
OUTPUT_TEX = os.path.join(BASE_DIR, "tables.tex")

DATASETS = ["reuters", "agnews", "20newsgroups", "dbpedia14"]

pattern = {
    "dataset": re.compile(r"Dataset:\s*(.*)"),
    "inlier": re.compile(r"Inlier class:\s*(.*)"),
    "model": re.compile(r"AD model:\s*(.*)"),
    "auc": re.compile(r"AUC:\s*([\d.]+)\s*±\s*([\d.]+)")
}

# results[dataset][inlier_class][model] = (mean, std)
results = defaultdict(lambda: defaultdict(dict))

def load_all_results():
    """Lit results.txt dans chaque dossier dataset."""
    for ds in DATASETS:
        folder = os.path.join(BASE_DIR, ds)
        filepath = os.path.join(folder, "results.txt")

        if not os.path.exists(filepath):
            print(f"Aucun fichier trouvé pour {ds}: {filepath}")
            continue

        with open(filepath, "r") as f:
            block = {"dataset": None, "inlier": None, "model": None, "auc": None}

            for line in f:
                line = line.strip()

                if m := pattern["dataset"].search(line):
                    block["dataset"] = m.group(1).strip()

                if m := pattern["inlier"].search(line):
                    block["inlier"] = m.group(1).strip()

                if m := pattern["model"].search(line):
                    block["model"] = m.group(1).strip()

                if m := pattern["auc"].search(line):
                    block["auc"] = (float(m.group(1)), float(m.group(2)))

                if line.startswith("========================================"):

                    if block["dataset"] and block["inlier"] and block["model"] and block["auc"]:
                        ds_name = block["dataset"]
                        ic = block["inlier"]
                        model = block["model"]
                        auc = block["auc"]

                        results[ds_name][ic][model] = auc

                    block = {"dataset": None, "inlier": None, "model": None, "auc": None}

MODEL_ORDER = [
    "ocsvm",
    "AE",
    "RSRAE",
    "CVDD",
    "FATE",
    "TCCM",
    "flow-matching"
]

MODEL_LATEX = {
    "ocsvm": "OCSVM",
    "AE": "AE",
    "RSRAE": "RSRAE",
    "CVDD": "CVDD",
    "FATE": "FATE",
    "TCCM": "TCCM",
    "flow-matching": "\\textbf{BasicFM}"
}

def fmt(mean, std):
    return f"{mean:.4f} $\\pm$ {std:.4f}"

def get_best_two(values):
    """values est une liste de floats, certains peuvent être None."""
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(valid) < 2:
        return None, None

    valid_sorted = sorted(valid, key=lambda x: x[1], reverse=True)

    best = valid_sorted[0][0]
    second = valid_sorted[1][0]
    return best, second

def generate_global_table(results):
    datasets = list(results.keys())

    global_means = []
    for model in MODEL_ORDER:
        row = []
        for ds in datasets:
            vals = [results[ds][ic][model] for ic in results[ds] if model in results[ds][ic]]
            if vals:
                means = [v[0] for v in vals]
                row.append(float(np.mean(means)))
            else:
                row.append(None)
        global_means.append(row)
        
    best_idx = {}
    sec_idx = {}
    for col in range(len(datasets)):
        column_vals = [global_means[row][col] for row in range(len(MODEL_ORDER))]
        best, second = get_best_two(column_vals)
        best_idx[col] = best
        sec_idx[col] = second


    latex = []
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\caption{AUC on All datasets}")
    latex.append("\\begin{tabular}{l" + "c"*len(datasets) + "}")
    latex.append("\\toprule")
    latex.append("Model & " + " & ".join([ds.capitalize() for ds in datasets]) + " \\\\")
    latex.append("\\midrule")

    for r, model in enumerate(MODEL_ORDER):
        row = [MODEL_LATEX[model]]
        for c, ds in enumerate(datasets):
            mean_val = global_means[r][c]
            if mean_val is not None:
                vals = [results[ds][ic][model] for ic in results[ds] if model in results[ds][ic]]
                if vals:
                    stds = [v[1] for v in vals]
                    mean_std = float(np.mean(stds))
                else:
                    mean_std = 0.0

                base = fmt(mean_val, mean_std)

                if r == best_idx[c]:
                    base = f"\\textbf{{{base}}}"
                elif r == sec_idx[c]:
                    base = f"\\underline{{{base}}}"
                row.append(base)
            else:
                row.append("0.0000 $\\pm$ 0.0000")
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\label{tab:ad_results}")
    latex.append("\\end{table}\n")

    return "\n".join(latex)

def generate_dataset_tables(results):

    latex_all = []

    for ds in results:
        inlier_classes = list(results[ds].keys())
        auc_matrix = []
        for model in MODEL_ORDER:
            row_vals = []
            for ic in inlier_classes:
                if model in results[ds][ic]:
                    row_vals.append(results[ds][ic][model][0])
                else:
                    row_vals.append(None)
            auc_matrix.append(row_vals)


        best_idx = {}
        sec_idx = {}
        for col in range(len(inlier_classes)):
            column_vals = [auc_matrix[row][col] for row in range(len(MODEL_ORDER))]
            best, second = get_best_two(column_vals)
            best_idx[col] = best
            sec_idx[col] = second

        latex = []
        latex.append("\\begin{table}[H]")
        latex.append("\\centering")
        latex.append(f"\\caption{{AUC on {ds.capitalize()}}}")
        latex.append("\\resizebox{1.1\\textwidth}{!}{")
        latex.append("\\begin{tabular}{l" + "c"*len(inlier_classes) + "}")
        latex.append("\\toprule")
        latex.append("Model & " + " & ".join(inlier_classes) + " \\\\")
        latex.append("\\midrule")

        for r, model in enumerate(MODEL_ORDER):

            row = [MODEL_LATEX[model]]
            

            for c, ic in enumerate(inlier_classes):

                if model in results[ds][ic]:
                    mean, std = results[ds][ic][model]
                    base = fmt(mean, std)

                    if r == best_idx[c]:
                        base = f"\\textbf{{{base}}}"
                    elif r == sec_idx[c]:
                        base = f"\\underline{{{base}}}"

                    row.append(base)
                else:
                    row.append("0.0000 $\\pm$ 0.0000")

            latex.append(" & ".join(row) + " \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("}")
        latex.append(f"\\label{{tab:{ds}}}")
        latex.append("\\end{table}\n")


        latex_all.append("\n".join(latex))

    return "\n\n".join(latex_all)

def create_tables():

    print("Lecture des fichiers dans chaque dataset...")
    load_all_results()

    global_table = generate_global_table(results)
    dataset_tables = generate_dataset_tables(results)

    if os.path.exists(OUTPUT_TEX):
        os.remove(OUTPUT_TEX)

    with open(OUTPUT_TEX, "w") as f:
        f.write(global_table)
        f.write("\n\n")
        f.write(dataset_tables)

    print(f"Fichier LaTeX généré : {OUTPUT_TEX}")


def save_results(
    dataset_name,
    inlier_topic,
    type_emb,
    ad_model,
    auc_mean, ap_mean, fpr_mean,
    auc_std=None, ap_std=None, fpr_std=None,
    train_time=None,             
    output_dir=BASE_DIR,            
    filename="results.txt",
    overwrite='smart'
):  
    dataset_folder = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    filepath = os.path.join(dataset_folder, filename)

    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_content = f.read()

    pattern = (
        rf"Dataset:\s*{re.escape(dataset_name)}\s*"
        rf"Inlier class:\s*{re.escape(inlier_topic)}\s*"
        rf"Embedding type:\s*{re.escape(type_emb)}\s*"
        rf"AD model:\s*{re.escape(ad_model)}"
    )

    def fmt(mean, std):
        """Formate 'mean ± std' si std est fourni, sinon seulement mean."""
        return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"

    time_str = f"{train_time:.2f} sec" if train_time is not None else "N/A"

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

    match = re.search(pattern, existing_content)

    if match:
        old_block_pattern = (
            r"========================================\n"
            + pattern +
            r".*?========================================\n\n"
        )
        old_block = re.search(old_block_pattern, existing_content, flags=re.DOTALL)

        if old_block:
            old_block = old_block.group(0)
            old_auc_match = re.search(r"AUC:\s*([\d.]+)", old_block)
            old_auc = float(old_auc_match.group(1)) if old_auc_match else -1
        else:
            old_auc = -1

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
            print(f"Résultats mis à jour pour ({dataset_name}, {inlier_topic}, {type_emb}, {ad_model}).")
        else:
            print(f"Résultats existants non modifiés pour ({dataset_name}, {inlier_topic}, {type_emb}, {ad_model}).")
            return

    else:
        existing_content += new_block
        print(f"Nouveaux résultats ajoutés pour ({dataset_name}, {inlier_topic}, {type_emb}, {ad_model}).")

    with open(filepath, "w") as f:
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

def load_hyperparams(dataset_name, inlier_topic, file_path):
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

    # Hyperparams regex
    hyper_pattern = re.compile(r"(\w+)\s*:\s*(.*)")

    for block in blocks:
        # Vérifier dataset_name et inlier_topic
        ds = ds_pattern.search(block)
        it = inlier_pattern.search(block)

        if not ds or not it:
            continue

        if ds.group(1).strip() == dataset_name and it.group(1).strip() == inlier_topic:
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
