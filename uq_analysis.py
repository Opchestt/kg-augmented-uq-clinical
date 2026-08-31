import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import warnings

# Suppress deprecation and other warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Sliding Window Attention.*")

def parse_args():
    parser = argparse.ArgumentParser(description="UQ Results Analysis with Qwen-72B Alignment")
    parser.add_argument("--limit_index", type=int, default=None, 
                        help="Max index to process. Set to limiting integer for fast testing (e.g. 50).")
    parser.add_argument("--reset", action="store_true", 
                        help="If set, delete existing checkpoint files and restart from scratch.")
    parser.add_argument("--ignore_expansion", action="store_true",
                        help="If set, globally filter out all diseases with is_expanded=True from evaluation.")
    parser.add_argument("--dataset", type=str, choices=['mimic', 'pmc', 'mtsamples'], default='mimic', help="Choose dataset to process")
    parser.add_argument("--model", type=str, choices=['llama', 'qwen', 'mistral', 'gemma'], default='llama', help="Choose model: llama, qwen, mistral, gemma")
    return parser.parse_args()

def load_and_preprocess(path, name):
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower()
        if 'uq' in df.columns:
            df = df.rename(columns={'uq': f'uq_{name}'})
        elif 'ccp_score' in df.columns:
            df = df.rename(columns={'ccp_score': f'uq_{name}'})
        df['index'] = df['index'].astype(int)
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return pd.DataFrame()

def get_data_for_mode(mode, dataset, model_choice, limit_index, ignore_expansion=False):
    if dataset == 'mimic':
        if model_choice == 'qwen':
            std_filename = f'standard_sampling_qwen_{mode}.csv'
        elif model_choice == 'mistral':
            std_filename = f'standard_sampling_mistral_{mode}.csv'
        elif model_choice == 'gemma':
            std_filename = f'standard_sampling_gemma_{mode}.csv'
        else:
            std_filename = f'standard_sampling_{mode}.csv'
    else:
        if model_choice == 'qwen':
            std_filename = f'standard_sampling_{dataset}_qwen_{mode}.csv'
        elif model_choice == 'mistral':
            std_filename = f'standard_sampling_{dataset}_mistral_{mode}.csv'
        elif model_choice == 'gemma':
            std_filename = f'standard_sampling_{dataset}_gemma_{mode}.csv'
        else:
            std_filename = f'standard_sampling_{dataset}_{mode}.csv'
        
    df_std = load_and_preprocess(std_filename, 'std')
    
    base_suffix = f"{mode}" if dataset == "mimic" else f"{dataset}_{mode}"
    if model_choice == 'qwen':
        comb_suffix = base_suffix + "_qwen"
    elif model_choice == 'mistral':
        comb_suffix = base_suffix + "_mistral"
    elif model_choice == 'gemma':
        comb_suffix = base_suffix + "_gemma"
    else:
        comb_suffix = base_suffix
    combined_path = f'combined_uq_{comb_suffix}.csv'
    df_combined = pd.DataFrame()
    if os.path.exists(combined_path):
        df_combined = pd.read_csv(combined_path)
        df_combined.columns = df_combined.columns.str.lower()
        if 'index' in df_combined.columns:
            df_combined['index'] = df_combined['index'].astype(int)
            
        if ignore_expansion and 'is_expanded' in df_combined.columns:
            print(f"Applying ignore_expansion filter: removing rows where is_expanded == True for {mode}")
            df_combined = df_combined[df_combined['is_expanded'] != True].copy()
        
    if not df_combined.empty:
        exp_col = ['is_expanded'] if 'is_expanded' in df_combined.columns else []
        df_ccp = df_combined[['index', 'disease', 'uq_ccp'] + exp_col].copy() if 'uq_ccp' in df_combined.columns else pd.DataFrame()
        df_kg = df_combined[['index', 'disease', 'uq_kg'] + exp_col].copy() if 'uq_kg' in df_combined.columns else pd.DataFrame()
        df_pure = df_combined[['index', 'disease', 'uq_pure'] + exp_col].copy() if 'uq_pure' in df_combined.columns else pd.DataFrame()
        df_kg_only = df_combined[['index', 'disease', 'kg_only'] + exp_col].copy() if 'kg_only' in df_combined.columns else pd.DataFrame()
        df_seq_prob = df_combined[['index', 'disease', 'uq_seq_prob'] + exp_col].copy() if 'uq_seq_prob' in df_combined.columns else pd.DataFrame()
        df_neg_ent = df_combined[['index', 'disease', 'uq_neg_ent'] + exp_col].copy() if 'uq_neg_ent' in df_combined.columns else pd.DataFrame()
        df_max_logprob = df_combined[['index', 'disease', 'uq_max_logprob'] + exp_col].copy() if 'uq_max_logprob' in df_combined.columns else pd.DataFrame()
        df_min_logprob = df_combined[['index', 'disease', 'uq_min_logprob'] + exp_col].copy() if 'uq_min_logprob' in df_combined.columns else pd.DataFrame()
        df_rrf = df_combined[['index', 'disease', 'uq_rrf'] + exp_col].copy() if 'uq_rrf' in df_combined.columns else pd.DataFrame()
    else:
        df_ccp = pd.DataFrame(columns=['index', 'disease', 'uq_ccp'])
        df_kg = pd.DataFrame(columns=['index', 'disease', 'uq_kg'])
        df_pure = pd.DataFrame(columns=['index', 'disease', 'uq_pure'])
        df_kg_only = pd.DataFrame(columns=['index', 'disease', 'kg_only'])
        df_seq_prob = pd.DataFrame(columns=['index', 'disease', 'uq_seq_prob'])
        df_neg_ent = pd.DataFrame(columns=['index', 'disease', 'uq_neg_ent'])
        df_max_logprob = pd.DataFrame(columns=['index', 'disease', 'uq_max_logprob'])
        df_min_logprob = pd.DataFrame(columns=['index', 'disease', 'uq_min_logprob'])
        df_rrf = pd.DataFrame(columns=['index', 'disease', 'uq_rrf'])
        
    if limit_index is not None:
        df_std = df_std[df_std['index'] <= limit_index] if not df_std.empty else df_std
        df_ccp = df_ccp[df_ccp['index'] <= limit_index] if not df_ccp.empty else df_ccp
        df_kg = df_kg[df_kg['index'] <= limit_index] if not df_kg.empty else df_kg
        df_pure = df_pure[df_pure['index'] <= limit_index] if not df_pure.empty else df_pure
        df_kg_only = df_kg_only[df_kg_only['index'] <= limit_index] if not df_kg_only.empty else df_kg_only
        df_seq_prob = df_seq_prob[df_seq_prob['index'] <= limit_index] if not df_seq_prob.empty else df_seq_prob
        df_neg_ent = df_neg_ent[df_neg_ent['index'] <= limit_index] if not df_neg_ent.empty else df_neg_ent
        df_max_logprob = df_max_logprob[df_max_logprob['index'] <= limit_index] if not df_max_logprob.empty else df_max_logprob
        df_min_logprob = df_min_logprob[df_min_logprob['index'] <= limit_index] if not df_min_logprob.empty else df_min_logprob
        df_rrf = df_rrf[df_rrf['index'] <= limit_index] if not df_rrf.empty else df_rrf
        
    all_indices = set()
    for df in [df_std, df_ccp, df_kg, df_pure, df_kg_only, df_seq_prob, df_neg_ent, df_max_logprob, df_min_logprob, df_rrf]:
        if not df.empty and 'index' in df.columns:
            all_indices.update(df['index'].unique())
            
    all_indices = sorted(list(all_indices))
    return df_std, df_ccp, df_kg, df_pure, df_kg_only, df_seq_prob, df_neg_ent, df_max_logprob, df_min_logprob, df_rrf, all_indices

def get_canonical_name(disease_name, context_list, model, tokenizer):
    disease_name_clean = str(disease_name).lower().strip()
    context_clean_map = {str(n).lower().strip(): n for n in context_list}
    if disease_name_clean in context_clean_map:
        return context_clean_map[disease_name_clean]
    
    candidates_str = ", ".join([f"'{n}'" for n in context_list])
    prompt = f"""You are an expert clinical coder and medical terminology specialist. 
Your task is to match a generated medical concept to a standardized list of diseases.

Target Concept: '{disease_name}'
Candidate List: {candidates_str}

CRITICAL RULES:
1. FOCUS ON THE CORE DISEASE: You MUST focus purely on the core medical condition.
2. SYNONYM & HIERARCHY MATCHING: If the core medical condition of the Target Concept matches, is a direct synonym of, or is a broader/narrower term within the EXACT SAME specific disease family as a Candidate, it is a valid match (e.g., "Diabetes" matches "Type 2 Diabetes Mellitus", "Depression" matches "Major Depressive Disorder").
3. STRICTLY NO CAUSE-EFFECT OR SYMPTOM MATCHING: You MUST NOT match a disease with its symptoms, clinical signs, underlying causes, or downstream complications. "Ascites" is a symptom, NOT "Liver Cirrhosis". "Confusion" is a symptom, NOT "Dementia".
4. BEWARE OF LEXICAL TRAPS (NO PROCEDURES/SIGNS): Just because words share a root does NOT make them a match. A clinical sign or lab status is NOT a disease. (e.g., "Parkinsonian gait" is a sign, NOT "Parkinson's disease". "Ascitic fluid analysis" is a procedure/lab, NOT the disease "Ascites").
5. SINGLE BEST MATCH: If multiple candidates are valid matches, select the single most accurate one.
6. STRICT OUTPUT FORMAT: Output ONLY the exact string from the Candidate List. Return NOTHING ELSE. No explanations, no formatting. If there is absolutely no match, output exactly 'None'.

EXAMPLES:
- Target: "Diabetes" | Candidates: ["Type 2 Diabetes Mellitus", "Hypertension"] 
  -> Output: Type 2 Diabetes Mellitus
- Target: "Parkinsonian gait" | Candidates: ["Parkinson's disease", "Essential tremor"] 
  -> Output: None
- Target: "Confusion" | Candidates: ["Dementia", "Alzheimer's Disease"] 
  -> Output: None
- Target: "Ascitic fluid analysis pending" | Candidates: ["Ascites", "Liver Cirrhosis"] 
  -> Output: None
- Target: "CHF" | Candidates: ["Chronic Heart Failure", "Kidney Disease"] 
  -> Output: Chronic Heart Failure

Output:"""
    
    messages = [
        {"role": "system", "content": "You align medical terms accurately."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
        
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    response_clean = response.strip("'").strip('"')
    response_lower = response_clean.lower()
    
    normalized_candidates = {c.lower(): c for c in context_list}
    if response_lower in normalized_candidates:
        return normalized_candidates[response_lower]
    return disease_name 

def evaluate_mode(mode, dataset, model_choice, limit_index, model, tokenizer, reset=False, ignore_expansion=False):
    print(f"\n--- STARTING EVALUATION FOR MODE: {mode.upper()} DATASET: {dataset.upper()} MODEL: {model_choice.upper()} ---")
    df_std, df_ccp, df_kg, df_pure, df_kg_only, df_seq_prob, df_neg_ent, df_max_logprob, df_min_logprob, df_rrf, all_indices = get_data_for_mode(mode, dataset, model_choice, limit_index, ignore_expansion)
    mode_str = f"{mode}" if dataset == "mimic" else f"{dataset}_{mode}"
    if model_choice == 'qwen':
        mode_str += "_qwen"
    elif model_choice == 'mistral':
        mode_str += "_mistral"
    elif model_choice == 'gemma':
        mode_str += "_gemma"
    
    if not all_indices:
        print(f"No data found for mode: {mode}")
        return [], 0.0
    
    metrics_ccp, metrics_kg, metrics_pure, metrics_kg_only = [], [], [], []
    metrics_seq_prob, metrics_neg_ent, metrics_max_logprob, metrics_min_logprob, metrics_rrf = [], [], [], [], []
    alignment_logs = []
    overlap_ratios = []
    
    # 斷點續傳邏輯
    processed_indices = set()
    metrics_file = f'note_metrics_{mode_str}.csv'
    checkpoint_files = [
        metrics_file,
        f'note_metrics_kg_{mode_str}.csv',
        f'note_metrics_ccp_{mode_str}.csv',
        f'note_metrics_pure_{mode_str}.csv',
        f'note_metrics_kg_only_{mode_str}.csv',
        f'note_metrics_seq_prob_{mode_str}.csv',
        f'note_metrics_neg_ent_{mode_str}.csv',
        f'note_metrics_max_logprob_{mode_str}.csv',
        f'note_metrics_min_logprob_{mode_str}.csv',
        f'note_metrics_rrf_{mode_str}.csv',
        f'alignment_logs_{mode_str}.csv',
        f'overlap_ratios_{mode_str}.csv'
    ]
    
    if reset:
        print("Reset flag is true. Deleting existing checkpoint files...")
        for f in checkpoint_files:
            if os.path.exists(f):
                os.remove(f)
                
    elif not ignore_expansion and os.path.exists(metrics_file):
        try:
            print(f"Found existing metrics file for {mode_str}, loading progress...")
            old_metrics = pd.read_csv(metrics_file)
            if 'index' in old_metrics.columns:
                processed_indices = set(old_metrics['index'].unique())
            
            # 讀取先前的紀錄
            metrics_kg = pd.read_csv(f'note_metrics_kg_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_kg_{mode_str}.csv') else []
            metrics_ccp = pd.read_csv(f'note_metrics_ccp_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_ccp_{mode_str}.csv') else []
            metrics_pure = pd.read_csv(f'note_metrics_pure_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_pure_{mode_str}.csv') else []
            metrics_kg_only = pd.read_csv(f'note_metrics_kg_only_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_kg_only_{mode_str}.csv') else []
            metrics_seq_prob = pd.read_csv(f'note_metrics_seq_prob_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_seq_prob_{mode_str}.csv') else []
            metrics_neg_ent = pd.read_csv(f'note_metrics_neg_ent_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_neg_ent_{mode_str}.csv') else []
            metrics_max_logprob = pd.read_csv(f'note_metrics_max_logprob_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_max_logprob_{mode_str}.csv') else []
            metrics_min_logprob = pd.read_csv(f'note_metrics_min_logprob_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_min_logprob_{mode_str}.csv') else []
            metrics_rrf = pd.read_csv(f'note_metrics_rrf_{mode_str}.csv').to_dict('records') if os.path.exists(f'note_metrics_rrf_{mode_str}.csv') else []
            
            if os.path.exists(f'alignment_logs_{mode_str}.csv'):
                alignment_logs = pd.read_csv(f'alignment_logs_{mode_str}.csv').to_dict('records')
            if os.path.exists(f'overlap_ratios_{mode_str}.csv'):
                overlap_ratios = pd.read_csv(f'overlap_ratios_{mode_str}.csv').to_dict('records')
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    indices_to_process = [idx for idx in all_indices if idx not in processed_indices]
    print(f"Total indices: {len(all_indices)}, Already processed: {len(processed_indices)}, Remaining: {len(indices_to_process)}")

    for i, idx in enumerate(tqdm(indices_to_process, desc=f"Processing {mode_str} Notes")):
        sub_std = df_std[df_std['index'] == idx].copy() if not df_std.empty and 'index' in df_std.columns else pd.DataFrame()
        sub_ccp = df_ccp[df_ccp['index'] == idx].copy() if not df_ccp.empty and 'index' in df_ccp.columns else pd.DataFrame()
        sub_kg = df_kg[df_kg['index'] == idx].copy() if not df_kg.empty and 'index' in df_kg.columns else pd.DataFrame()
        sub_pure = df_pure[df_pure['index'] == idx].copy() if not df_pure.empty and 'index' in df_pure.columns else pd.DataFrame()
        sub_kg_only = df_kg_only[df_kg_only['index'] == idx].copy() if not df_kg_only.empty and 'index' in df_kg_only.columns else pd.DataFrame()
        sub_seq_prob = df_seq_prob[df_seq_prob['index'] == idx].copy() if not df_seq_prob.empty and 'index' in df_seq_prob.columns else pd.DataFrame()
        sub_neg_ent = df_neg_ent[df_neg_ent['index'] == idx].copy() if not df_neg_ent.empty and 'index' in df_neg_ent.columns else pd.DataFrame()
        sub_max_logprob = df_max_logprob[df_max_logprob['index'] == idx].copy() if not df_max_logprob.empty and 'index' in df_max_logprob.columns else pd.DataFrame()
        sub_min_logprob = df_min_logprob[df_min_logprob['index'] == idx].copy() if not df_min_logprob.empty and 'index' in df_min_logprob.columns else pd.DataFrame()
        sub_rrf = df_rrf[df_rrf['index'] == idx].copy() if not df_rrf.empty and 'index' in df_rrf.columns else pd.DataFrame()
        
        # 直接濾除 standard sampling 中 score <= 0.05 的結果
        if sub_std.empty:
            continue
            
        sub_std = sub_std[sub_std['uq_std'] > 0.05].copy()
        
        if sub_std.empty:
            continue
            
        std_diseases = sub_std['disease'].dropna().unique().tolist()
        if not std_diseases: continue
            
        # ====== Optimize Alignment: Align unique diseases once per note ======
        all_method_dfs = [sub_ccp, sub_kg, sub_pure, sub_kg_only, sub_seq_prob, sub_neg_ent, sub_max_logprob, sub_min_logprob, sub_rrf]
        unique_diseases = set()
        for df in all_method_dfs:
            if not df.empty and 'disease' in df.columns:
                unique_diseases.update(df['disease'].dropna().tolist())
                
        alignment_cache = {}
        for d in unique_diseases:
            canonical = get_canonical_name(d, std_diseases, model, tokenizer)
            alignment_cache[d] = canonical
            
            is_matched = canonical in std_diseases or str(canonical).lower() in [t.lower() for t in std_diseases]
            if canonical != d and str(canonical).lower() != str(d).lower():
                alignment_logs.append({'index': idx, 'original': d, 'aligned': canonical, 'status': 'Aligned'})
            elif canonical == d and not is_matched:
                alignment_logs.append({'index': idx, 'original': d, 'aligned': canonical, 'status': 'Unmatched'})

        def align_and_aggregate(df_source, uq_col):
            if df_source.empty:
                df_source['disease_aligned'] = []
                return df_source
            
            df_source['disease_aligned'] = df_source['disease'].map(alignment_cache)
            agg_funcs = {uq_col: 'max'}
            if 'is_expanded' in df_source.columns:
                agg_funcs['is_expanded'] = 'min'  # If any maps to False, it means it originated from LM, so min() correctly resolves to False.
            return df_source.groupby('disease_aligned').agg(agg_funcs).reset_index()

        sub_ccp = align_and_aggregate(sub_ccp, "uq_ccp")
        sub_kg = align_and_aggregate(sub_kg, "uq_kg")
        sub_pure = align_and_aggregate(sub_pure, "uq_pure")
        sub_kg_only = align_and_aggregate(sub_kg_only, "kg_only")
        sub_seq_prob = align_and_aggregate(sub_seq_prob, "uq_seq_prob")
        sub_neg_ent = align_and_aggregate(sub_neg_ent, "uq_neg_ent")
        sub_max_logprob = align_and_aggregate(sub_max_logprob, "uq_max_logprob")
        sub_min_logprob = align_and_aggregate(sub_min_logprob, "uq_min_logprob")
        sub_rrf = align_and_aggregate(sub_rrf, "uq_rrf")
        
        sub_std['disease_aligned'] = sub_std['disease']
        sub_std = sub_std.groupby('disease_aligned')['uq_std'].max().reset_index()

        # ====== Calculate Generation vs Sampling Overlap ======
        gen_diseases = set(sub_ccp['disease_aligned'].dropna()) if not sub_ccp.empty else set()
        std_set = set(sub_std['disease_aligned'].dropna()) if not sub_std.empty else set()
        
        if len(gen_diseases) > 0:
            inter_count = len(gen_diseases.intersection(std_set))
            overlap_ratios.append({
                'index': idx,
                'overlap_ratio': inter_count / len(gen_diseases)
            })
        else:
            overlap_ratios.append({
                'index': idx,
                'overlap_ratio': 0.0
            })
        # ====================================================

        def calc_note_metrics(sub_std_df, sub_pred, pred_col, allow_expanded=False):
            valid_sub = sub_pred.copy()
            if 'is_expanded' in valid_sub.columns and not allow_expanded:
                valid_sub = valid_sub[valid_sub['is_expanded'] == False]
                
            # Strictly Isolated Evaluation: Build universal set ONLY for this method
            method_diseases = set(sub_std_df['disease_aligned'].dropna())
            if not valid_sub.empty:
                method_diseases |= set(valid_sub['disease_aligned'].dropna())
                
            df_method = pd.DataFrame({'disease': list(method_diseases)})
            df_method = df_method.merge(sub_std_df[['disease_aligned', 'uq_std']], left_on='disease', right_on='disease_aligned', how='left').drop(columns=['disease_aligned'])
            
            if not valid_sub.empty:
                df_method = df_method.merge(valid_sub[['disease_aligned', pred_col]], left_on='disease', right_on='disease_aligned', how='left').drop(columns=['disease_aligned'])
            else:
                df_method[pred_col] = np.nan
                
            fill_val = -100.0 if pred_col in ['uq_neg_ent', 'uq_max_logprob', 'uq_min_logprob'] else 0.0
            df_method['uq_std'] = df_method['uq_std'].fillna(0.0)
            df_method[pred_col] = df_method[pred_col].fillna(fill_val)
            
            y_true = df_method['uq_std'].values
            y_pred = df_method[pred_col].values

            if len(y_true) < 2: return None 
            if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
                k_corr = 0.0
            else:
                k_corr, _ = kendalltau(y_true, y_pred)
            
            # NDCG
            try:
                ndcg_5 = ndcg_score([y_true], [y_pred], k=5)
                ndcg_10 = ndcg_score([y_true], [y_pred], k=10)
                std_len = len(sub_std_df)
                ndcg_std = ndcg_score([y_true], [y_pred], k=std_len) if std_len > 0 else 0.0
            except ValueError:
                ndcg_5, ndcg_10, ndcg_std = 0.0, 0.0, 0.0
                
            # --- Metrics without padding (MRR, RBO, Precision, Recall) ---
            true_sorted_diseases = sub_std_df.sort_values('uq_std', ascending=False)['disease_aligned'].tolist() if not sub_std_df.empty else []
            pred_sorted_diseases = valid_sub.sort_values(pred_col, ascending=False)['disease_aligned'].tolist() if not valid_sub.empty else []
            
            # MRR
            mrr = 0.0
            true_set = set(true_sorted_diseases)
            for rank, d in enumerate(pred_sorted_diseases, 1):
                if d in true_set:
                    mrr = 1.0 / rank
                    break
                    
            # RBO p=0.9
            def calculate_rbo(l1, l2, p=0.9):
                if not len(l1) or not len(l2): return 0.0
                set1, set2 = set(), set()
                score = 0.0
                max_depth = max(len(l1), len(l2))
                for j in range(max_depth):
                    if j < len(l1): set1.add(l1[j])
                    if j < len(l2): set2.add(l2[j])
                    intersection = len(set1 & set2)
                    score += (intersection / (j + 1)) * (p ** j)
                return score * (1 - p)
                
            rbo_score = calculate_rbo(true_sorted_diseases, pred_sorted_diseases, p=0.9)
                
            # Precision & Recall @ K
            pred_top_5 = set(pred_sorted_diseases[:5])
            pred_top_10 = set(pred_sorted_diseases[:10])
            
            p5 = len(true_set & pred_top_5) / len(pred_top_5) if pred_top_5 else 0.0
            r5 = len(true_set & pred_top_5) / len(true_set) if true_set else 0.0
            p10 = len(true_set & pred_top_10) / len(pred_top_10) if pred_top_10 else 0.0
            r10 = len(true_set & pred_top_10) / len(true_set) if true_set else 0.0

            return {
                'index': idx, 
                'NDCG@5': ndcg_5,
                'NDCG@10': ndcg_10,
                'NDCG@std': ndcg_std,
                'RBO': rbo_score,
                'MRR': mrr,
                'Recall@5': r5,
                'Recall@10': r10,
                'Precision@5': p5,
                'Precision@10': p10,
                'Kendall_tau_b': k_corr
            }

        m_ccp = calc_note_metrics(sub_std, sub_ccp, 'uq_ccp', allow_expanded=False)
        if m_ccp: metrics_ccp.append(m_ccp)
        m_kg = calc_note_metrics(sub_std, sub_kg, 'uq_kg', allow_expanded=True)
        if m_kg: metrics_kg.append(m_kg)
        m_pure = calc_note_metrics(sub_std, sub_pure, 'uq_pure', allow_expanded=False)
        if m_pure: metrics_pure.append(m_pure)
        m_kg_only = calc_note_metrics(sub_std, sub_kg_only, 'kg_only', allow_expanded=True)
        if m_kg_only: metrics_kg_only.append(m_kg_only)
        m_seq_prob = calc_note_metrics(sub_std, sub_seq_prob, 'uq_seq_prob', allow_expanded=False)
        if m_seq_prob: metrics_seq_prob.append(m_seq_prob)
        m_neg_ent = calc_note_metrics(sub_std, sub_neg_ent, 'uq_neg_ent', allow_expanded=False)
        if m_neg_ent: metrics_neg_ent.append(m_neg_ent)
        m_max_logprob = calc_note_metrics(sub_std, sub_max_logprob, 'uq_max_logprob', allow_expanded=False)
        if m_max_logprob: metrics_max_logprob.append(m_max_logprob)
        m_min_logprob = calc_note_metrics(sub_std, sub_min_logprob, 'uq_min_logprob', allow_expanded=False)
        if m_min_logprob: metrics_min_logprob.append(m_min_logprob)
        m_rrf = calc_note_metrics(sub_std, sub_rrf, 'uq_rrf', allow_expanded=True)
        if m_rrf: metrics_rrf.append(m_rrf)

        # 儲存進度 (每跑 10 筆寫入一次)
        if (i + 1) % 10 == 0 and not ignore_expansion:
            if alignment_logs:
                pd.DataFrame(alignment_logs).to_csv(f'alignment_logs_{mode_str}.csv', index=False)
            if metrics_kg:
                pd.DataFrame(metrics_kg).to_csv(f'note_metrics_{mode_str}.csv', index=False)
                pd.DataFrame(metrics_kg).to_csv(f'note_metrics_kg_{mode_str}.csv', index=False)
            if metrics_ccp:
                pd.DataFrame(metrics_ccp).to_csv(f'note_metrics_ccp_{mode_str}.csv', index=False)
            if metrics_pure:
                pd.DataFrame(metrics_pure).to_csv(f'note_metrics_pure_{mode_str}.csv', index=False)
            if metrics_kg_only:
                pd.DataFrame(metrics_kg_only).to_csv(f'note_metrics_kg_only_{mode_str}.csv', index=False)
            if metrics_seq_prob:
                pd.DataFrame(metrics_seq_prob).to_csv(f'note_metrics_seq_prob_{mode_str}.csv', index=False)
            if metrics_neg_ent:
                pd.DataFrame(metrics_neg_ent).to_csv(f'note_metrics_neg_ent_{mode_str}.csv', index=False)
            if metrics_max_logprob:
                pd.DataFrame(metrics_max_logprob).to_csv(f'note_metrics_max_logprob_{mode_str}.csv', index=False)
            if metrics_min_logprob:
                pd.DataFrame(metrics_min_logprob).to_csv(f'note_metrics_min_logprob_{mode_str}.csv', index=False)
            if metrics_rrf:
                pd.DataFrame(metrics_rrf).to_csv(f'note_metrics_rrf_{mode_str}.csv', index=False)
            if overlap_ratios:
                pd.DataFrame(overlap_ratios).to_csv(f'overlap_ratios_{mode_str}.csv', index=False)

    if alignment_logs and not ignore_expansion:
        df_logs = pd.DataFrame(alignment_logs)
        df_logs.to_csv(f'alignment_logs_{mode_str}.csv', index=False)

    summary = []
    def append_summary(metrics_list, method_name):
        if not metrics_list: return
        df_res = pd.DataFrame(metrics_list)
        summary.append({
            'Mode': mode_str.upper(),
            'Method': method_name,
            'N': len(df_res),
            'NDCG@5_Mean': df_res['NDCG@5'].mean(),
            'NDCG@10_Mean': df_res['NDCG@10'].mean(),
            'NDCG@std_Mean': df_res['NDCG@std'].mean(),
            'RBO_Mean': df_res['RBO'].mean(),
            'MRR_Mean': df_res['MRR'].mean(),
            'Recall@5_Mean': df_res['Recall@5'].mean(),
            'Recall@10_Mean': df_res['Recall@10'].mean(),
            'Precision@5_Mean': df_res['Precision@5'].mean(),
            'Precision@10_Mean': df_res['Precision@10'].mean(),
            'Kendall_tau_b_Mean': df_res['Kendall_tau_b'].mean(),
            'Kendall_tau_b_SD': df_res['Kendall_tau_b'].std()
        })

    append_summary(metrics_pure, "Method Pure Logits")
    append_summary(metrics_ccp, "Method CCP")
    append_summary(metrics_kg, "Method KG-Augmented")
    append_summary(metrics_kg_only, "Method KG Only")
    append_summary(metrics_seq_prob, "Method Seq Prob")
    append_summary(metrics_neg_ent, "Method Neg Ent")
    append_summary(metrics_max_logprob, "Method Max Logprob")
    append_summary(metrics_min_logprob, "Method Min Logprob")
    append_summary(metrics_rrf, "Method RRF")
    
    # Save note-level metrics for downstream analysis
    if not ignore_expansion:
        if metrics_kg:
            pd.DataFrame(metrics_kg).to_csv(f'note_metrics_{mode_str}.csv', index=False)
            pd.DataFrame(metrics_kg).to_csv(f'note_metrics_kg_{mode_str}.csv', index=False)
        if metrics_ccp:
            pd.DataFrame(metrics_ccp).to_csv(f'note_metrics_ccp_{mode_str}.csv', index=False)
        if metrics_pure:
            pd.DataFrame(metrics_pure).to_csv(f'note_metrics_pure_{mode_str}.csv', index=False)
        if metrics_kg_only:
            pd.DataFrame(metrics_kg_only).to_csv(f'note_metrics_kg_only_{mode_str}.csv', index=False)
        if metrics_seq_prob:
            pd.DataFrame(metrics_seq_prob).to_csv(f'note_metrics_seq_prob_{mode_str}.csv', index=False)
        if metrics_neg_ent:
            pd.DataFrame(metrics_neg_ent).to_csv(f'note_metrics_neg_ent_{mode_str}.csv', index=False)
        if metrics_max_logprob:
            pd.DataFrame(metrics_max_logprob).to_csv(f'note_metrics_max_logprob_{mode_str}.csv', index=False)
        if metrics_min_logprob:
            pd.DataFrame(metrics_min_logprob).to_csv(f'note_metrics_min_logprob_{mode_str}.csv', index=False)
        if metrics_rrf:
            pd.DataFrame(metrics_rrf).to_csv(f'note_metrics_rrf_{mode_str}.csv', index=False)
    
    if overlap_ratios:
        df_overlaps = pd.DataFrame(overlap_ratios)
        if not ignore_expansion:
            df_overlaps.to_csv(f"overlap_ratios_{mode_str}.csv", index=False)
        mean_overlap = df_overlaps['overlap_ratio'].mean()
    else:
        mean_overlap = 0.0

    return summary, mean_overlap


def align_and_aggregate_intra(df_source, target_list, uq_col, model, tokenizer):
    if df_source.empty:
        df_source['disease_aligned'] = []
        return df_source
    
    aligned_names = []
    for d in df_source['disease']:
        canonical = get_canonical_name(d, target_list, model, tokenizer)
        aligned_names.append(canonical)
        
    df_source['disease_aligned'] = aligned_names
def load_for_intra(path, target_col, mode_suffix):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if 'uq' in df.columns and target_col == 'uq_std':
        df = df.rename(columns={'uq': target_col})
    if target_col in df.columns:
        res = df[['index', 'disease', target_col]].copy()
        res = res.rename(columns={target_col: f'uq_{mode_suffix}'})
        res['index'] = res['index'].astype(int)
        return res
    return pd.DataFrame()

def compare_full_vs_sliced(method_name, full_path, sliced_path, target_col, limit_index, model, tokenizer):
    print(f"\n--- Comparing {method_name} (Full vs Sliced) ---")
    df_full = load_for_intra(full_path, target_col, 'full')
    df_sliced = load_for_intra(sliced_path, target_col, 'sliced')
    
    if df_full.empty or df_sliced.empty:
        print(f"Missing data for {method_name} comparison.")
        return None
        
    if limit_index is not None:
         df_full = df_full[df_full['index'] <= limit_index]
         df_sliced = df_sliced[df_sliced['index'] <= limit_index]
         
    indices = sorted(list(set(df_full['index'].unique()) | set(df_sliced['index'].unique())))
    diffs, sliced_only_uqs = [], []
    
    for idx in tqdm(indices, desc=f"Intra-Method {method_name}"):
        sub_full = df_full[df_full['index'] == idx].copy() if not df_full.empty and 'index' in df_full.columns else pd.DataFrame()
        sub_sliced = df_sliced[df_sliced['index'] == idx].copy() if not df_sliced.empty and 'index' in df_sliced.columns else pd.DataFrame()
        
        if sub_full.empty and sub_sliced.empty: continue
            
        full_diseases = sub_full['disease'].dropna().unique().tolist() if not sub_full.empty else []
        sub_sliced_aligned = align_and_aggregate_intra(sub_sliced, full_diseases, 'uq_sliced', model, tokenizer)
        
        if not sub_full.empty:
            sub_full['disease_aligned'] = sub_full['disease']
            sub_full_agg = sub_full.groupby('disease_aligned')['uq_full'].max().reset_index()
        else:
            sub_full_agg = pd.DataFrame(columns=['disease_aligned', 'uq_full'])
            
        merged = pd.merge(sub_full_agg, sub_sliced_aligned, on='disease_aligned', how='outer')
        
        common = merged.dropna(subset=['uq_full', 'uq_sliced']).copy()
        if not common.empty:
            common['diff'] = common['uq_full'] - common['uq_sliced']
            diffs.extend(common['diff'].tolist())
            
        sliced_only = merged[merged['uq_full'].isna() & merged['uq_sliced'].notna()]
        if not sliced_only.empty:
            sliced_only_uqs.extend(sliced_only['uq_sliced'].tolist())
            
    return {
        'Method': method_name,
        'Intersection_Cases': len(diffs),
        'Mean_Diff(UQ_Full - UQ_Sliced)': np.mean(diffs) if diffs else 0,
        'Sliced_Only_Cases': len(sliced_only_uqs),
        'Mean_UQ_Sliced_Only': np.mean(sliced_only_uqs) if sliced_only_uqs else 0
    }

def main():
    args = parse_args()
    limit_index = args.limit_index
    reset = args.reset
    ignore_expansion = args.ignore_expansion
    dataset = args.dataset
    print(f"Configurations:\n- LIMIT_INDEX: {limit_index}\n- RESET: {reset}\n- IGNORE_EXPANSION: {ignore_expansion}\n- DATASET: {dataset}\n")

    model_choice = args.model

    out_dir = f"results/{dataset}"
    if model_choice == 'qwen':
        out_dir = f"results/{dataset}/qwen"
    elif model_choice == 'mistral':
        out_dir = f"results/{dataset}/mistral"
    elif model_choice == 'gemma':
        out_dir = f"results/{dataset}/gemma"
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    print(f"Loading Model: {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 1. Evaluate Correlation (Spearman, Pearson, MAE) for Full and Sliced modes
    modes = ["full", "sliced"]
    correlation_summaries = []
    overlap_results = []
    
    for mode in modes:
        mode_summary, mean_overlap = evaluate_mode(mode, dataset, model_choice, limit_index, model, tokenizer, reset=reset, ignore_expansion=ignore_expansion)
        correlation_summaries.extend(mode_summary)
        
        current_overlap = {
            'Mode': mode.upper(),
            'Generation_Overlap_Ratio': mean_overlap
        }
        overlap_results.append(current_overlap)
        
        print(f"\n--- Correlation Results ({mode.upper()}) ---")
        if mode_summary:
            print(pd.DataFrame(mode_summary).to_string(index=False))
        else:
            print("No summary data.")
            
        print(f"\n--- Generation vs Sampling Overlap ({mode.upper()}) ---")
        print(pd.DataFrame([current_overlap]).to_string(index=False))
        
    df_corr_summary = pd.DataFrame(correlation_summaries)
    
    exp_suffix = "_no_exp" if ignore_expansion else ""
    summary_filename = f"correlation_summary_results_{dataset}{exp_suffix}.csv"
    if model_choice == 'qwen':
        summary_filename = f"correlation_summary_results_{dataset}_qwen{exp_suffix}.csv"
    elif model_choice == 'mistral':
        summary_filename = f"correlation_summary_results_{dataset}_mistral{exp_suffix}.csv"
    elif model_choice == 'gemma':
        summary_filename = f"correlation_summary_results_{dataset}_gemma{exp_suffix}.csv"
        
    df_corr_summary.to_csv(summary_filename, index=False)

    # 2. Intra-method Comparison
    # intra_summaries = []
    
    # methods_to_compare = [
    #     ("Standard Sampling", "standard_sampling_full.csv", "standard_sampling_sliced.csv", "uq_std"),
    #     ("Method CCP", "combined_uq_full.csv", "combined_uq_sliced.csv", "uq_ccp"),
    #     ("Method KG", "combined_uq_full.csv", "combined_uq_sliced.csv", "uq_kg"),
    #     ("Pure Logits", "combined_uq_full.csv", "combined_uq_sliced.csv", "uq_pure"),
    #     ("KG Only", "combined_uq_full.csv", "combined_uq_sliced.csv", "kg_only")
    # ]
    
    # for m_name, p_full, p_sliced, t_col in methods_to_compare:
    #     res = compare_full_vs_sliced(m_name, p_full, p_sliced, t_col, limit_index, model, tokenizer)
    #     if res:
    #         intra_summaries.append(res)
            
    # if intra_summaries:
    #     df_intra_summary = pd.DataFrame(intra_summaries)
    #     df_intra_summary.to_csv("intra_method_summary_results.csv", index=False)
    #     print("\n--- Intra-method Comparison Results ---")
    #     print(df_intra_summary.to_string(index=False))

if __name__ == "__main__":
    main()