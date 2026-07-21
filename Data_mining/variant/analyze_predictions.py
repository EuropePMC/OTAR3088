import os
import re
import ast
import pandas as pd

def clean_and_normalize(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r'[()\[\]\s\u2009]', '', text)
    text = text.replace("->", ">").replace("−>", ">")
    return text

def classify_results(preds, golds):
    tp = []
    partial = []
    miss = []
    fp = []
    
    cleaned_golds = []
    for g in golds:
        cleaned_golds.append({
            "original": g,
            "clean_txt": clean_and_normalize(g.get("text", "")),
            "matched": False
        })
        
    cleaned_preds = []
    for p in preds:
        cleaned_preds.append({
            "original": p,
            "clean_txt": clean_and_normalize(p.get("text", "")),
            "matched": False
        })
        
    # 1. True Positives
    for p in cleaned_preds:
        for g in cleaned_golds:
            if g["matched"]:
                continue
            if p["clean_txt"] == g["clean_txt"] and p["original"].get("label") == g["original"].get("label"):
                p["matched"] = True
                g["matched"] = True
                tp.append(f"{p['original'].get('text')} ({p['original'].get('label')})")
                break
                
    # 2. Partials
    for p in cleaned_preds:
        if p["matched"]:
            continue
        for g in cleaned_golds:
            if g["matched"]:
                continue
            p_text = p["clean_txt"]
            g_text = g["clean_txt"]
            if p_text and g_text and (p_text in g_text or g_text in p_text):
                p["matched"] = True
                g["matched"] = True
                partial.append(f"{p['original'].get('text')} -> {g['original'].get('text')} (Label: {p['original'].get('label')} vs {g['original'].get('label')})")
                break
                
    # 3. Misses
    for g in cleaned_golds:
        if not g["matched"]:
            miss.append(f"{g['original'].get('text')} ({g['original'].get('label')})")
            
    # 4. False Positives
    for p in cleaned_preds:
        if not p["matched"]:
            fp.append(f"{p['original'].get('text')} ({p['original'].get('label')})")
            
    return tp, partial, miss, fp

def parse_diagnostics_blocks(file_path):
    blocks = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    raw_blocks = content.split("Question ID: ")
    for rb in raw_blocks[1:]:
        lines = rb.strip().split("\n")
        qid = lines[0].strip()
        
        sentence_lines = []
        in_sentence = False
        parsed_llm = []
        parsed_gt = []
        
        for line in lines:
            if line.startswith("Raw LLM Response:"):
                in_sentence = False
            elif line.startswith("Parsed LLM Response:"):
                try:
                    parsed_llm = ast.literal_eval(line.replace("Parsed LLM Response:", "").strip()).get("entities", [])
                except:
                    parsed_llm = []
            elif line.startswith("Parsed GT Response:"):
                try:
                    parsed_gt = ast.literal_eval(line.replace("Parsed GT Response:", "").strip()).get("entities", [])
                except:
                    parsed_gt = []
            elif in_sentence:
                sentence_lines.append(line.strip())
            elif line.startswith("Sentence:"):
                if "Annotate the genetic variants in the following sentence" in line:
                    continue
                in_sentence = True
                sentence_lines.append(line.replace("Sentence:", "").strip())
                
        sentence_text = " ".join(sentence_lines).strip()
        blocks.append({
            "qid": qid,
            "sentence": sentence_text,
            "preds": parsed_llm,
            "golds": parsed_gt
        })
    return blocks

def analyze_predictions(output_dir="output"):
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found.")
        return
        
    sentence_rows = []
    entity_rows = []
    
    for file in sorted(os.listdir(output_dir)):
        if file.endswith("_diagnostics.txt"):
            file_path = os.path.join(output_dir, file)
            
            # Extract run metadata
            parts = file.replace("_diagnostics.txt", "").split("_")
            if len(parts) >= 3:
                run_type = "_".join(parts[:2])
                model_name = "_".join(parts[2:])
            else:
                run_type = parts[0]
                model_name = "_".join(parts[1:])
                
            model = model_name.replace("_", ":")
            condition = run_type.replace("_", " ").title()
            
            blocks = parse_diagnostics_blocks(file_path)
            for b in blocks:
                preds_list = [f"{p.get('text')} ({p.get('label')})" for p in b["preds"]]
                golds_list = [f"{g.get('text')} ({g.get('label')})" for g in b["golds"]]
                
                tp, partial, miss, fp = classify_results(b["preds"], b["golds"])
                
                # Format a result summary string
                result_parts = []
                if tp: result_parts.append("TP")
                if partial: result_parts.append("Partial")
                if miss: result_parts.append("Miss")
                if fp: result_parts.append("FP")
                result_summary = ", ".join(result_parts) if result_parts else "TP"
                
                # 1. Sentence-level row
                sentence_rows.append({
                    "sentence": b["sentence"],
                    "model": model,
                    "condition": condition,
                    "predictions": preds_list,
                    "ground truth": golds_list,
                    "result": result_summary
                })
                
                # 2. Entity-level rows (exploded details)
                for g in b["golds"]:
                    g_txt = clean_and_normalize(g.get("text"))
                    matched_type = "Miss"
                    matched_pred_txt = ""
                    matched_pred_lbl = ""
                    
                    for p in b["preds"]:
                        p_txt = clean_and_normalize(p.get("text"))
                        if p_txt == g_txt and p.get("label") == g.get("label"):
                            matched_type = "TP"
                            matched_pred_txt = p.get("text")
                            matched_pred_lbl = p.get("label")
                            break
                        elif p_txt and g_txt and (p_txt in g_txt or g_txt in p_txt):
                            matched_type = "Partial"
                            matched_pred_txt = p.get("text")
                            matched_pred_lbl = p.get("label")
                            
                    entity_rows.append({
                        "sentence": b["sentence"],
                        "model": model,
                        "condition": condition,
                        "gold_text": g.get("text"),
                        "gold_label": g.get("label"),
                        "pred_text": matched_pred_txt or "None",
                        "pred_label": matched_pred_lbl or "None",
                        "result": matched_type
                    })
                    
                # Find false positives (preds not matched to any gold)
                for p in b["preds"]:
                    p_txt = clean_and_normalize(p.get("text"))
                    has_match = False
                    for g in b["golds"]:
                        g_txt = clean_and_normalize(g.get("text"))
                        if p_txt == g_txt or (p_txt and g_txt and (p_txt in g_txt or g_txt in p_txt)):
                            has_match = True
                            break
                    if not has_match:
                        entity_rows.append({
                            "sentence": b["sentence"],
                            "model": model,
                            "condition": condition,
                            "gold_text": "None",
                            "gold_label": "None",
                            "pred_text": p.get("text"),
                            "pred_label": p.get("label"),
                            "result": "FP"
                        })
                        
    df_sentence = pd.DataFrame(sentence_rows)
    df_entity = pd.DataFrame(entity_rows)
    
    sentence_path = os.path.join(output_dir, "sentence_predictions_analysis.tsv")
    df_sentence.to_csv(sentence_path, sep="\t", index=False)
    print(f"Saved sentence-level predictions analysis to: {sentence_path}")
    
    entity_path = os.path.join(output_dir, "entity_predictions_analysis.tsv")
    df_entity.to_csv(entity_path, sep="\t", index=False)
    print(f"Saved entity-level predictions analysis to: {entity_path}")
    
    return df_sentence

if __name__ == "__main__":
    analyze_predictions()
