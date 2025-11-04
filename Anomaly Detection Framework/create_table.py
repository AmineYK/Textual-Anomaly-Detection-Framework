import re
from collections import defaultdict

# Fichier source
input_file = "Results/results.txt"
output_file = "Results/results.tex"


# structure pour stocker les résultats
# results[dataset][inlier][embedding][model] = (mean, std)
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

# lecture du fichier
with open(input_file, "r") as f:
    content = f.read()

# pattern regex pour chaque bloc, capture mean et std
pattern = re.compile(
    r"Dataset:\s*(.*?)\n"
    r"Inlier class:\s*(.*?)\n"
    r"Embedding type:\s*(.*?)\n"
    r"AD model:\s*(.*?)\n"
    r".*?AUC:\s*([\d.]+)\s*±\s*([\d.]+)",
    re.DOTALL
)

for match in pattern.finditer(content):
    dataset, inlier, embedding, model, mean_auc, std_auc = match.groups()
    results[dataset][inlier][embedding][model] = (float(mean_auc), float(std_auc))

# génération du fichier .tex
with open(output_file, "w") as f:
    f.write("\\begin{table}[ht]\n\\centering\n")
    
    # header
    f.write("\\begin{tabular}{l|cc|cc}\n")
    f.write("\\hline\n")
    f.write(" & \\multicolumn{2}{c|}{GloVe} & \\multicolumn{2}{c}{FastText} \\\\\n")
    f.write("Class & OC-SVM & CVDD & OC-SVM & CVDD \\\\\n")
    f.write("\\hline\n")
    
    for dataset, inliers in results.items():
        f.write(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{dataset}}}}} \\\\\n")
        for inlier, emb_dict in inliers.items():
            glove_ocsvm = emb_dict.get("glove", {}).get("ocsvm", ("-", "-"))
            glove_cvdd  = emb_dict.get("glove", {}).get("cvdd", ("-", "-"))
            ft_ocsvm    = emb_dict.get("fasttext", {}).get("ocsvm", ("-", "-"))
            ft_cvdd     = emb_dict.get("fasttext", {}).get("cvdd", ("-", "-"))
            
            # format AUC ± std avec 2 décimales
            glove_ocsvm_str = f"{glove_ocsvm[0]:.4f} ± {glove_ocsvm[1]:.4f}" if glove_ocsvm != ("-", "-") else "-"
            glove_cvdd_str  = f"{glove_cvdd[0]:.4f} ± {glove_cvdd[1]:.4f}" if glove_cvdd != ("-", "-") else "-"
            ft_ocsvm_str    = f"{ft_ocsvm[0]:.4f} ± {ft_ocsvm[1]:.4f}" if ft_ocsvm != ("-", "-") else "-"
            ft_cvdd_str     = f"{ft_cvdd[0]:.4f} ± {ft_cvdd[1]:.4f}" if ft_cvdd != ("-", "-") else "-"
            
            f.write(f"{inlier} & {glove_ocsvm_str} & {glove_cvdd_str} & {ft_ocsvm_str} & {ft_cvdd_str} \\\\\n")
    
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.write("\\caption{AUC des modèles AD pour différents embeddings et datasets, avec écart type.}\n")
    f.write("\\label{tab:ad_results}\n")
    f.write("\\end{table}\n")

print(f"Table LaTeX générée dans {output_file}")