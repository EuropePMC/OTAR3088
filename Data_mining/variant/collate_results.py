import os
import re
import ast
import pandas as pd

def clean_and_normalize(text):
    text = text.lower().strip()
    # Remove parentheses, brackets, whitespace, and common non-breaking space characters
    text = re.sub(r'[()\[\]\s\u2009]', '', text)
    # Normalize delimiters
    text = text.replace("->", ">").replace("−>", ">")
    return text

def calculate_fuzzy_metrics(pred_entities, gold_entities):
    if not pred_entities and not gold_entities:
        return 1.0, 1.0, 1.0
    if not pred_entities or not gold_entities:
        return 0.0, 0.0, 0.0
        
    cleaned_golds = [clean_and_normalize(e['text'] if isinstance(e, dict) else e.text) for e in gold_entities]
    cleaned_preds = [clean_and_normalize(e['text'] if isinstance(e, dict) else e.text) for e in pred_entities]
    
    # Remove empty strings to avoid trivial substring matches
    cleaned_golds = [g for g in cleaned_golds if g]
    cleaned_preds = [p for p in cleaned_preds if p]
    
    if not cleaned_preds and not cleaned_golds:
        return 1.0, 1.0, 1.0
    if not cleaned_preds or not cleaned_golds:
        return 0.0, 0.0, 0.0

    matched_golds = 0
    for g in cleaned_golds:
        # Check if gold standard string is inside a predicted string, or vice versa
        if any(g in p or p in g for p in cleaned_preds):
            matched_golds += 1
            
    matched_preds = 0
    for p in cleaned_preds:
        # Check if predicted string is inside a gold standard string, or vice versa
        if any(g in p or p in g for g in cleaned_golds):
            matched_preds += 1
            
    precision = matched_preds / len(cleaned_preds)
    recall = matched_golds / len(cleaned_golds)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def parse_diagnostics_file(path):
    exact_f1_scores = []
    fuzzy_f1_scores = []
    fuzzy_recalls = []
    pass_count = 0
    total_count = 0
    
    with open(path, "r", encoding="utf-8") as f:
        current_preds = []
        current_golds = []
        
        for line in f:
            m_f1 = re.search(r"F1-Score: (.*)", line)
            if m_f1:
                f1_str = m_f1.group(1).strip()
                f1_val = float(f1_str) if f1_str != "None" else 0.0
                exact_f1_scores.append(f1_val)
                if f1_val >= 0.9999: # Exact match
                    pass_count += 1
                total_count += 1
                
                # Now calculate fuzzy metrics for the current question
                _, recall, f1 = calculate_fuzzy_metrics(current_preds, current_golds)
                fuzzy_f1_scores.append(f1)
                fuzzy_recalls.append(recall)
                
                # Reset for next question
                current_preds = []
                current_golds = []
                continue
                
            m_pred = re.search(r"Parsed LLM Response: (.*)", line)
            if m_pred:
                try:
                    pred_dict = ast.literal_eval(m_pred.group(1).strip())
                    current_preds = pred_dict.get("entities", [])
                except Exception:
                    current_preds = []
                    
            m_gold = re.search(r"Parsed GT Response: (.*)", line)
            if m_gold:
                try:
                    gold_dict = ast.literal_eval(m_gold.group(1).strip())
                    current_golds = gold_dict.get("entities", [])
                except Exception:
                    current_golds = []
                
    if total_count == 0:
        return None
        
    avg_exact_f1 = sum(exact_f1_scores) / len(exact_f1_scores)
    avg_fuzzy_f1 = sum(fuzzy_f1_scores) / len(fuzzy_f1_scores)
    avg_fuzzy_recall = sum(fuzzy_recalls) / len(fuzzy_recalls)
    pass_rate = pass_count / total_count
    
    return {
        "total": total_count,
        "avg_exact_f1": avg_exact_f1,
        "avg_fuzzy_f1": avg_fuzzy_f1,
        "avg_fuzzy_recall": avg_fuzzy_recall,
        "pass_rate": pass_rate
    }

def generate_comments(results):
    comments = []
    
    # Parse numbers to float for calculations
    model_groups = {}
    for r in results:
        model = r["Model"]
        if model not in model_groups:
            model_groups[model] = {}
        
        try:
            exact_f1 = float(r["Avg F1 (Exact)"].replace("%", "")) / 100.0
            fuzzy_f1 = float(r["Avg F1 (Fuzzy)"].replace("%", "")) / 100.0
            recall = float(r["Coverage (Recall)"].replace("%", "")) / 100.0
            model_groups[model][r["Run Type"]] = {
                "exact_f1": exact_f1,
                "fuzzy_f1": fuzzy_f1,
                "recall": recall
            }
        except (ValueError, KeyError):
            continue
            
    comments.append("### Key Observations\n")
    
    # 1. Boundary mismatch Analysis (Exact vs Fuzzy)
    comments.append("#### 1. Exact F1 vs. Fuzzy F1 (Boundary Mismatch Penalty)")
    for model, runs in sorted(model_groups.items()):
        for run_type, metrics in sorted(runs.items()):
            diff = (metrics["fuzzy_f1"] - metrics["exact_f1"]) * 100.0
            comments.append(
                f"- **{model} ({run_type})**: Fuzzy F1 is **{diff:+.2f}%** relative to Exact F1 (Fuzzy: {metrics['fuzzy_f1']:.2%}, Exact: {metrics['exact_f1']:.2%}, Coverage/Recall: {metrics['recall']:.2%})."
            )
            
    # 2. Compare Zero Shot vs Skill Guided (Prompt Impact)
    has_prompt_comparison = False
    for model, runs in sorted(model_groups.items()):
        if "Zero Shot" in runs and "Skill Guided" in runs:
            if not has_prompt_comparison:
                comments.append("\n#### 2. Prompt Engineering Impact (Zero-Shot vs. Skill-Guided)")
                has_prompt_comparison = True
            diff_exact = (runs["Skill Guided"]["exact_f1"] - runs["Zero Shot"]["exact_f1"]) * 100.0
            diff_fuzzy = (runs["Skill Guided"]["fuzzy_f1"] - runs["Zero Shot"]["fuzzy_f1"]) * 100.0
            comments.append(
                f"- **{model}**: Skill-guided instructions changed Exact F1 by **{diff_exact:+.2f}%** (Zero-Shot: {runs['Zero Shot']['exact_f1']:.2%}, Skill-Guided: {runs['Skill Guided']['exact_f1']:.2%}) and Fuzzy F1 by **{diff_fuzzy:+.2f}%**."
            )
                
    # 3. Model Scale Comparison (14B vs 7B)
    has_scale_comparison = False
    for run_type in ["Zero Shot", "Skill Guided"]:
        models_with_run = [m for m in model_groups if run_type in model_groups[m]]
        for m1 in sorted(models_with_run):
            if "7b" in m1.lower():
                # Search for corresponding 14b model
                m2_target = m1.lower().replace("7b", "14b")
                matching_14b = [m for m in models_with_run if m.lower() == m2_target]
                if matching_14b:
                    m14b = matching_14b[0]
                    if not has_scale_comparison:
                        comments.append("\n#### 3. Model Scale Impact (14B vs. 7B)")
                        has_scale_comparison = True
                    diff_f1_exact = (model_groups[m14b][run_type]["exact_f1"] - model_groups[m1][run_type]["exact_f1"]) * 100.0
                    diff_f1_fuzzy = (model_groups[m14b][run_type]["fuzzy_f1"] - model_groups[m1][run_type]["fuzzy_f1"]) * 100.0
                    comments.append(
                        f"- **{run_type}**: The 14B model (`{m14b}`) achieved a **{diff_f1_exact:+.2f}% Exact F1 difference** and **{diff_f1_fuzzy:+.2f}% Fuzzy F1 difference** over the 7B model (`{m1}`)."
                    )
                    
    return "\n".join(comments)

def main():
    output_dir = "output"
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found.")
        return
        
    results = []
    for file in sorted(os.listdir(output_dir)):
        if file.endswith("_diagnostics.txt"):
            path = os.path.join(output_dir, file)
            # Filename format: {run_type}_{model_name}_diagnostics.txt
            # e.g., zero_shot_qwen2.5_14b_diagnostics.txt
            parts = file.replace("_diagnostics.txt", "").split("_")
            if len(parts) >= 3:
                run_type = "_".join(parts[:2])
                model_name = "_".join(parts[2:])
            else:
                run_type = parts[0]
                model_name = "_".join(parts[1:])
                
            metrics = parse_diagnostics_file(path)
            if metrics:
                results.append({
                    "Model": model_name.replace("_", ":"),
                    "Run Type": run_type.replace("_", " ").title(),
                    "Sentences": metrics["total"],
                    "Avg F1 (Exact)": f"{metrics['avg_exact_f1']:.2%}",
                    "Avg F1 (Fuzzy)": f"{metrics['avg_fuzzy_f1']:.2%}",
                    "Coverage (Recall)": f"{metrics['avg_fuzzy_recall']:.2%}",
                    "Exact Pass Rate": f"{metrics['pass_rate']:.2%}"
                })
                
    if not results:
        print("No diagnostics logs found in output/ directory.")
        return
        
    def to_markdown_table(data_list):
        if not data_list:
            return ""
        keys = list(data_list[0].keys())
        header = "| " + " | ".join(keys) + " |"
        separator = "| " + " | ".join(["---"] * len(keys)) + " |"
        rows = []
        for item in data_list:
            row = "| " + " | ".join(str(item[k]) for k in keys) + " |"
            rows.append(row)
        return "\n".join([header, separator] + rows)
        
    markdown_table = to_markdown_table(results)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON REPORT")
    print("=" * 80)
    print(markdown_table)
    print("=" * 80)
    
    # Generate dynamic observations
    dynamic_commentary = generate_comments(results)
    
    # Save markdown report
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Benchmark Comparison Report\n\n")
        f.write(markdown_table)
        f.write("\n\n## Performance Comments & Insights\n\n")
        f.write(dynamic_commentary)
        f.write("\n")
        
    print(f"\nSaved benchmark report to: {report_path}")

if __name__ == "__main__":
    main()
