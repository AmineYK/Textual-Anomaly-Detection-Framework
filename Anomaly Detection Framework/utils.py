import re
from collections import defaultdict
import numpy as np
import os

BASE_DIR = "/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results"
OUTPUT_TEX = os.path.join(BASE_DIR, "tables.tex")

DATASETS = ["reuters", "agnews", "20newsgroups"]

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
    "ae",
    "RSRAE",
    "CVDD",
    "FATE",
    "FM_AD_tab",
    "flow-matching"
]

MODEL_LATEX = {
    "ocsvm": "OCSVM",
    "ae": "AE",
    "RSRAE": "RSRAE",
    "CVDD": "CVDD",
    "FATE": "FATE",
    "FM_AD_tab": "FM\\_AD\\_tab",
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
    output_dir=BASE_DIR,   # BASE_DIR défini plus haut
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

    new_block = (
        "========================================\n"
        f"Dataset:        {dataset_name}\n"
        f"Inlier class:   {inlier_topic}\n"
        f"Embedding type: {type_emb}\n"
        f"AD model:       {ad_model}\n"
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

