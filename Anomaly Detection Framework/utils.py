
import re
import os

def save_results(args, auc_mean, ap_mean, fpr_mean, auc_std=None, ap_std=None, fpr_std=None,
                 output_dir="/home/2017025/ayouce01/Textual-Anomaly-Detection-Framework/Anomaly Detection Framework/Results",
                 filename="results.txt",
                 overwrite=None): 

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    existing_content = ""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            existing_content = f.read()

  
    pattern = (
        rf"Dataset:\s*{re.escape(args.dataset_name)}\s*"
        rf"Inlier class:\s*{re.escape(args.inlier_topic)}\s*"
        rf"Embedding type:\s*{re.escape(args.type_emb)}\s*"
        rf"AD model:\s*{re.escape(args.ad_model)}"
    )


    def fmt(mean, std):
        """Formate 'mean ± std' si std est fourni, sinon seulement mean."""
        return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"

    new_block = (
        "========================================\n"
        f"Dataset:        {args.dataset_name}\n"
        f"Inlier class:   {args.inlier_topic}\n"
        f"Embedding type: {args.type_emb}\n"
        f"AD model:       {args.ad_model}\n"
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
            existing_content = re.sub(old_block_pattern, new_block, existing_content, flags=re.DOTALL)
            print(f"Résultats mis à jour pour ({args.dataset_name}, {args.inlier_topic}, {args.type_emb}, {args.ad_model}).")
        else:
            print(f"Résultats existants non modifiés pour ({args.dataset_name}, {args.inlier_topic}, {args.type_emb}, {args.ad_model}).")
            return
    else:
        existing_content += new_block
        print(f"Nouveaux résultats ajoutés pour ({args.dataset_name}, {args.inlier_topic}, {args.type_emb}, {args.ad_model}).")

    with open(filepath, "w") as f:
        f.write(existing_content)
