import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import re
import gc
import json
import argparse
import torch
import pandas as pd
import numpy as np
import networkx as nx
import scipy.sparse as sp
import ctypes
import torch.nn.functional as F
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModel,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)

# --- Configuration ---
KG_PATH = os.environ.get("KG_PATH", "data/primeKG/kg.csv")
KGE_EMB_PATH = os.environ.get("KGE_EMB_PATH", "data/primeKG/primekg_rotate_dim256_embeddings.npy")
KGE_MAP_PATH = os.environ.get("KGE_MAP_PATH", "data/primeKG/primekg_rotate_dim256_mapping.csv")
DEFAULT_INPUT_CSV = 'data/MIMIC_notes_icd_01.csv'
DEFAULT_OUTPUT_CSV = 'combined_uq.csv'
OPTIONAL_STATS_CSV = 'combined_mapping_stats.csv'
RWR_STATS_CSV = 'rwr_node_stats.csv'
HF_TOKEN = os.environ.get("HF_TOKEN", "")

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    libc = ctypes.CDLL("libc.so.6")
except Exception:
    libc = None

def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Unified UQ Pipeline (Pure Logits + CCP + KG)")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=['full', 'sliced'], default='full')
    parser.add_argument("--top_k_ccp", type=int, default=5, help="Top-K for CCP alternatives")
    parser.add_argument("--alpha", type=float, default=0.25, help="KG-Augmented alpha")
    parser.add_argument("--tf_threshold_type", type=str, choices=['pure_logits', 'min_logprob', 'none'], default='min_logprob', help="Metric to threshold KG expanded targets")
    parser.add_argument("--tf_threshold_val", type=float, default=-12.6, help="Threshold value for KG expanded targets (e.g., -8.2 for min_logprob, 0.01 for pure_logits)")
    parser.add_argument("--top_k_kg", type=int, default=5, help="Top-K for KG expansion candidates")
    parser.add_argument("--restart_rate", type=float, default=0.2, help="Probability of restart in RWR")
    parser.add_argument("--dataset", type=str, choices=['mimic', 'pmc', 'mtsamples'], default='mimic', help="Choose dataset to process")
    parser.add_argument("--model", type=str, choices=['llama', 'qwen', 'mistral', 'gemma'], default='llama', help="Choose model: llama or qwen or mistral or gemma")
    return parser.parse_args()

# --- 1. Load PrimeKG ---
def load_primekg():
    print("Loading PrimeKG...")
    df_kg = pd.read_csv(KG_PATH, low_memory=False)
    G = nx.Graph()
    disease_dag = nx.DiGraph()
    
    valid_edges = df_kg.dropna(subset=['x_index', 'y_index'])
    
    # [修改] 更新 edge relations 與對應的 weight 權重分配
    keep_relations = [
        'phenotype present', 'indication', 'off-label use', 
        'parent-child', 'associated with', 'side effect', 'target', 'enzyme',
        'transporter', 'carrier'
    ]
    valid_edges = valid_edges[valid_edges['display_relation'].str.lower().isin(keep_relations)].copy()
    
    weight_map = {
        'phenotype present': 5.0, 
        'indication': 3.0, 
        'off-label use': 3.0,
        'parent-child': 1.0, 
        'associated with': 1.0, 
        'side effect': 1.0, 
        'target': 1.0,
        'enzyme': 1.0,
        'transporter': 1.0,
        'carrier': 1.0  
    }
    valid_edges['weight'] = valid_edges['display_relation'].str.lower().map(weight_map).fillna(1.0)
    
    # [1. 修改] Check parent-child relation to build Specificity DAG (Child -> Parent i.e. y -> x)
    pc_edges = valid_edges[valid_edges['display_relation'] == 'parent-child'].dropna(subset=['x_type', 'y_type'])
    inv_pc = list(zip(pc_edges['y_index'].astype(int), pc_edges['x_index'].astype(int)))
    disease_dag.add_edges_from(inv_pc)

    edges = list(zip(valid_edges['x_index'].astype(int), valid_edges['y_index'].astype(int), valid_edges['weight']))
    G.add_weighted_edges_from(edges)

    node_map, index_to_name, index_to_type, index_to_id = {}, {}, {}, {}
    nodes_x = df_kg[['x_index', 'x_id', 'x_type', 'x_name']].dropna(subset=['x_index']).rename(columns={'x_index':'index', 'x_name':'name', 'x_type':'type', 'x_id': 'id'})
    nodes_y = df_kg[['y_index', 'y_id', 'y_type', 'y_name']].dropna(subset=['y_index']).rename(columns={'y_index':'index', 'y_name':'name', 'y_type':'type', 'y_id': 'id'})
    all_nodes = pd.concat([nodes_x, nodes_y]).drop_duplicates(subset=['index'])
    
    for _, row in all_nodes.iterrows():
        idx = int(row['index'])
        name = str(row['name']).lower() if pd.notna(row['name']) else ""
        node_type = str(row['type'])
        node_id = str(row['id'])
        node_map[name], index_to_name[idx], index_to_type[idx], index_to_id[idx] = idx, name, node_type, node_id

    # [修改] 限制 expansion targets (disease nodes) 只包含 disease 與 effect/phenotype 兩種 node type
    disease_nodes = [idx for idx, t in index_to_type.items() if t in ['disease', 'effect/phenotype']]

    # [1. 修改] 計算 Inverted PageRank (-log(PR)) 作為 Specificity score，並包含所有 disease 節點避免孤立
    disease_dag.add_nodes_from(disease_nodes)
    pr_spec = nx.pagerank(disease_dag, alpha=0.85)
    node_specificity = {n: -np.log(pr) for n, pr in pr_spec.items()}
    
    print("Loading PrimeKG Embedding...")
    kge_embs = np.load(KGE_EMB_PATH, mmap_mode='r')
    kge_map_df = pd.read_csv(KGE_MAP_PATH)
    kge_map = {str(k): v for k, v in zip(kge_map_df['primekg_node_id'], kge_map_df['matrix_index'])}

    # Free huge dataframe memory
    del df_kg, valid_edges, all_nodes, pc_edges
    gc.collect()

    print("Pre-building Sparse Matrix for PageRank...")
    nodelist = list(G.nodes())
    node_index_map = {n: i for i, n in enumerate(nodelist)}
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=np.float32, weight='weight')
    out_degree = np.array(A.sum(axis=1), dtype=np.float32).flatten()
    
    with np.errstate(divide='ignore'):
        inv_degree = 1.0 / out_degree
    inv_degree[np.isinf(inv_degree)] = 0.0
    
    Q = sp.diags(inv_degree, format='csr') @ A
    is_dangling = np.where(out_degree == 0)[0]

    unweighted_degree = np.array([G.degree(n) for n in nodelist], dtype=np.float32)
    pr_comps = (Q, is_dangling, nodelist, node_index_map, unweighted_degree)
    valid_nodes = set(nodelist)
    del G, A
    gc.collect()

    return valid_nodes, node_specificity, index_to_name, index_to_type, index_to_id, disease_nodes, kge_embs, kge_map, pr_comps

# --- 2. Load Models & KG Embeddings ---
embed_tokenizer, embed_model = None, None
gen_tokenizer, gen_model = None, None
nli_tokenizer, nli_model, entailment_id = None, None, 1

def load_models(load_gen=True, model_choice='llama'):
    global embed_tokenizer, embed_model, gen_tokenizer, gen_model, nli_tokenizer, nli_model, entailment_id
    
    print("Loading embedding model (SapBERT)...")
    embed_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    embed_tokenizer = AutoTokenizer.from_pretrained(embed_name)
    embed_model = AutoModel.from_pretrained(embed_name, torch_dtype=torch.float16).to(device)
    embed_model.eval()

    if load_gen:
        if model_choice == 'llama':
            gen_model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_choice == 'qwen':
            gen_model_name = "Qwen/Qwen2.5-7B-Instruct"
        elif model_choice == 'mistral':
            gen_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        elif model_choice == 'gemma':
            gen_model_name = "google/gemma-3-4b-it"
            
        print(f"Loading Gen Model ({gen_model_name})...")
        
        is_gemma = (model_choice == 'gemma')
        
        # Gemma 3 4B + 4-bit 量化會產生 inf/nan logits，導致 CUDA assertion
        # Gemma 3 4B 夠小，直接用 bfloat16 載入即可（約 8GB VRAM）
        if is_gemma:
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": device,
                "trust_remote_code": True,
                "token": HF_TOKEN,
                "attn_implementation": "eager"
            }
        else:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4")
            kwargs = {
                "quantization_config": bnb_config,
                "device_map": device,
                "trust_remote_code": True,
                "token": HF_TOKEN
            }
            if model_choice == 'llama':
                kwargs["attn_implementation"] = "eager"
            
        gen_model = AutoModelForCausalLM.from_pretrained(gen_model_name, **kwargs)
        gen_model.eval()
        gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name, token=HF_TOKEN, padding_side="left")
        if gen_tokenizer.pad_token is None: gen_tokenizer.pad_token = gen_tokenizer.eos_token


    print("Loading NLI model...")
    nli_model_name = "pritamdeka/PubMedBERT-MNLI-MedNLI"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name, torch_dtype=torch.float16).to(device)
    nli_model.eval()
    if nli_model.config.label2id: entailment_id = nli_model.config.label2id.get('entailment', 1)

def get_embeddings(texts):
    def mean_pooling(mo, am):
        te = mo[0]
        mask = am.unsqueeze(-1).expand(te.size()).float()
        return torch.sum(te * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
    texts = [str(t).lower() for t in texts]
    inputs = embed_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad(): out = embed_model(**inputs)
    return mean_pooling(out, inputs['attention_mask'])

def precompute_kg_embeddings(index_to_name, index_to_type):
    print("Pre-computing KG Embeddings...")
    # [修改] 還原為全部節點，讓 Qwen 的 entity mapping 不受 node type 限制
    valid_indices = [idx for idx, t in index_to_type.items()]
    valid_names = [index_to_name[idx] for idx in valid_indices]
    kg_embeddings = []
    
    for i in tqdm(range(0, len(valid_names), 512)):
        kg_embeddings.append(get_embeddings(valid_names[i:i+512]).cpu().to(torch.float16))
        
    kg_embeddings = F.normalize(torch.cat(kg_embeddings, dim=0), p=2, dim=1).float()
    return valid_indices, valid_names, kg_embeddings

# --- 3. KG Mapping & Context ---
def map_to_kg(entity_list, kg_embs, valid_indices, valid_names, nli_model, nli_tokenizer, entailment_id, threshold=0.70, return_details=False):
    if not entity_list: return ([], []) if return_details else []
    try:
        embs = F.normalize(get_embeddings(entity_list).cpu(), p=2, dim=1)
    except: return ([], []) if return_details else []

    sim_matrix = torch.mm(embs, kg_embs.t())
    top_vals, top_idx = torch.topk(sim_matrix, k=5, dim=1)
    mapped = []
    details = []

    for i, entity in enumerate(entity_list):
        mapped_current = False
        for rank in range(5):
            val = top_vals[i, rank].item()
            if val < threshold: break
            cand_name = valid_names[top_idx[i, rank].item()]
            
            if check_nli_entailment(nli_model, nli_tokenizer, entailment_id, entity, cand_name):
                mapped.append(valid_indices[top_idx[i, rank].item()])
                details.append(f"{entity}  -> mapped ->  {cand_name} (Bi:{val:.2f})")
                mapped_current = True
                break
        
        if not mapped_current:
            details.append(f"{entity} -> UNMAPPED")
            
    return (list(set(mapped)), details) if return_details else list(set(mapped))

def get_rwr_personalization(valid_nodes, node_specificity, kge_embs, kge_map, start_nodes, index_to_id, alpha_w=1.0, beta_w=1.0):
    valid_start_nodes = [n for n in start_nodes if n in valid_nodes]
    if not valid_start_nodes: return {}, []
    
    # 1. Coherence Score (C)
    kge_vectors = []
    node_to_vec = {}
    for n in valid_start_nodes:
        # 使用 PrimeKG 的字串 ID 去查 kge_map
        mapped_id = index_to_id.get(n, str(n))
        if mapped_id in kge_map:
            mat_idx = kge_map[mapped_id]
            vec = kge_embs[mat_idx]
            # 修正：將 RotatE 的複數向量展開成 2 倍長度的實數向量，以計算正確的 Cosine Similarity
            vec = np.concatenate([np.real(vec), np.imag(vec)])
            kge_vectors.append(vec)
            node_to_vec[n] = vec
    
    centroid_norm = 0
    if kge_vectors:
        centroid = np.mean(kge_vectors, axis=0)
        centroid_norm = np.linalg.norm(centroid)
    
    stats = []
    personalization = {}
    for n in valid_start_nodes:
        # C 
        C = 1.0 # Default if no embedding
        if n in node_to_vec and centroid_norm > 0:
            v_norm = np.linalg.norm(node_to_vec[n])
            if v_norm > 0:
                cos_sim = np.dot(node_to_vec[n], centroid) / (v_norm * centroid_norm)
                C = (cos_sim + 1.0) / 2.0
                
        # S (Inverted PageRank specificity)
        S = node_specificity.get(n, 1.0)
        # N is no longer relevant, kept variable to avoid breaking stats logging
        N = 0
        
        # W
        W = (C ** alpha_w) * (S ** beta_w)
        personalization[n] = W
        stats.append({'node_index': n, 'C': C, 'S': S, 'W': W})
        
    total_w = sum(personalization.values())
    if total_w > 0:
        personalization = {k: v/total_w for k, v in personalization.items()}
    else:
        personalization = {n: 1.0/len(valid_start_nodes) for n in valid_start_nodes}
        
    for s in stats:
        s['prob'] = personalization[s['node_index']]
        
    return personalization, stats

def compute_rwr(G, personalization, target_idx, all_diseases, restart_rate=0.2):
    if not personalization or target_idx not in G: return 0.0
    try:
        pr = nx.pagerank(G, alpha=1-restart_rate, personalization=personalization, max_iter=50)
        target_score = pr.get(target_idx, 0.0)
        disease_vals = np.array([pr.get(x, 0.0) for x in all_diseases])
        if len(disease_vals) == 0: return 0.0
        return (disease_vals < target_score).mean()
    except: return 0.0

# --- 4. UQ Methods ---
def check_nli_entailment(nli_model, nli_tokenizer, entailment_id, text_center, text_candidate):
    if text_center.lower().strip() == text_candidate.lower().strip(): return True
    pa, pb = f"The patient is diagnosed with {text_center}.", f"The patient is diagnosed with {text_candidate}."
    hb, ha = pb, pa
    inputs = nli_tokenizer([pa, pb], [hb, ha], return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad(): probs = torch.softmax(nli_model(**inputs).logits, dim=-1)
    if probs[0, entailment_id].item() > 0.5 or probs[1, entailment_id].item() > 0.5: return True
    return False

def calculate_ccp(gen_model, tokenizer, prompt_ids, prefix_tokens, orig_name, start_logits, top_k):
    probs = F.softmax(start_logits, dim=-1)
    top_k_probs, top_k_ids = torch.topk(probs, k=top_k)
    base_input = torch.cat([prompt_ids, prefix_tokens.unsqueeze(0)], dim=1)
    
    ent_sum, tot_sum = 0.0, 0.0
    for rank in range(top_k):
        alt_id, alt_prob = top_k_ids[rank].item(), top_k_probs[rank].item()
        tot_sum += alt_prob
        
        curr_in = torch.cat([base_input, torch.tensor([[alt_id]], device=device)], dim=1)
        is_gemma = ('gemma' in gen_model.config.model_type.lower())
        ccp_kwargs = dict(max_new_tokens=20, do_sample=False, temperature=None, top_p=None,
                          repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                          eos_token_id=[tokenizer.eos_token_id, tokenizer.encode('\n')[-1]])
        if not is_gemma:
            ccp_kwargs['no_repeat_ngram_size'] = 3
        with torch.no_grad():
            out = gen_model.generate(curr_in, **ccp_kwargs)
        full_alt = tokenizer.decode(out[0, base_input.shape[1]:], skip_special_tokens=True).split('\n')[0].strip()
        
        # 釋放 CCP 生成的小 Tensors
        del curr_in, out
        
        if check_nli_entailment(nli_model, nli_tokenizer, entailment_id, orig_name, full_alt):
            ent_sum += alt_prob
            
    return ent_sum / tot_sum if tot_sum > 0 else 0.0

def combined_uq_pipeline(note_text, top_k_ccp, kg_embs, valid_idx, valid_names, valid_nodes, disease_nodes, alpha, pre_extracted_entities, node_specificity, primekg_embs, primekg_map, idx2id, idx2type, idx2name, pr_comps, tf_threshold_type='min_logprob', tf_threshold_val=-8.2, top_k_kg=10, restart_rate=0.2, idx_for_logging=0):
    # Truncate text for Llama
    tokens_scan = gen_tokenizer(note_text, add_special_tokens=False).input_ids
    if len(tokens_scan) > 7300: note_text = gen_tokenizer.decode(tokens_scan[-7300:])
        
    prompt_msgs = [
            {
                "role": "system",
                "content": "You are an expert clinical diagnostician. Your goal is to analyze medical notes to extract explicitly stated diagnoses and infer highly probable diagnoses based on current symptoms, findings, and active clinical presentation, without over-speculating."
            },
            {
                "role": "user",
                "content": f"""Analyze the following discharge summary and list all potential and confirmed disease diagnoses that are strongly supported by the patient's presentation.

            Please output the answer as a list in the following format:
            - [Disease name 1]
            - [Disease name 2]
            - ...

            IMPORTANT RULES:
            1. EXTRACT AND INFER: First, ensure you extract all explicitly stated, confirmed diagnoses from the text. Then, infer any other highly probable diagnoses logically supported by the presentation.
            2. Output ONLY the specific disease name. Each line MUST start with "- " (a hyphen followed by a space).
            - CORRECT: "- Coronary Artery Disease"
            - INCORRECT: "-Coronary Artery Disease" (Missing space)
            3. CANONICAL ENTITIES ONLY: Use atomic, standard clinical terminology (e.g., SNOMED or ICD style). Do NOT include surgical procedures, anatomical specificities, or status indicators (like "s/p", "status post").
            - Correct: "- Coronary Artery Disease"
            - Incorrect: "- CAD s/p 3 v CABG LIMA-LAD"
            4. CURRENT/ACTIVE PROBLEMS ONLY: Strictly exclude past medical history. Do NOT extract entities describing remote past events or starting with "History of...", "h/o", etc.
            5. STRICTLY NO EXPLANATIONS: Do NOT include reasoning, evidence, etiology, probabilities, or notes. NEVER use words like "due to", "secondary to", "cannot be ruled out", "suspected", or "likely".
            - Correct: "- Postural Orthostatic Tachycardia Syndrome"
            - Incorrect: "- Postural Orthostatic Tachycardia Syndrome (POTS) cannot be ruled out due to tachycarida despite adequate hydration"
            6. Do not answer same disease multiple times.
            7. CONSERVATIVE INFERENCE: Only list diseases that are highly probable.
            Discharge Summary:
            {note_text}"""
        }
    ]
    
    prompt_text = gen_tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    inputs = gen_tokenizer(prompt_text, return_tensors="pt").to(device)
    
    # 1. GENERATE
    # Gemma 3 不支援 no_repeat_ngram_size，會觸發 CUDA device-side assertion
    is_gemma = (gen_model.config.model_type == 'gemma3' or 'gemma' in gen_model.config.model_type.lower())
    gen_kwargs = dict(
        max_new_tokens=512, do_sample=False, temperature=None, top_p=None,
        repetition_penalty=1.2,
        output_scores=True, return_dict_in_generate=True, pad_token_id=gen_tokenizer.eos_token_id
    )
    if not is_gemma:
        gen_kwargs['no_repeat_ngram_size'] = 3
    with torch.no_grad():
        out = gen_model.generate(**inputs, **gen_kwargs)
        
    gen_tokens = out.sequences[0][inputs.input_ids.shape[1]:]
    
    # 列印原始生成的完整結果供觀察
    raw_gen_text = gen_tokenizer.decode(gen_tokens, skip_special_tokens=False)
    tqdm.write(f"\n[{'='*20} Index {idx_for_logging} 原始生成文字 {'='*20}]\n{raw_gen_text}\n{'='*60}\n")
    
    # [COMPRESSION 1] 立刻半精度化: 把 128k 維度的 logits 轉為 float16/bfloat16，直接省下一半的記憶體空間
    logits = torch.stack(out.scores, dim=0).squeeze(1).to(torch.float16)
    
    prompt_ids = inputs.input_ids.detach().clone()
    del inputs, out
    torch.cuda.empty_cache()
    
    # 2. EXTRACT KG CONTEXT (由 Step 1 提供，不用載入 Qwen)
    context_ents = pre_extracted_entities
    context_nodes, mapping_details = map_to_kg(context_ents, kg_embs, valid_idx, valid_names, nli_model, nli_tokenizer, entailment_id, return_details=True)
    
    tqdm.write(f"\n[Index {idx_for_logging}] 準備進行 KG 隨機遊走 (RWR)... 映射與計算全局 {len(context_nodes)} 個起點...")
    note_personalization, rwr_stats = get_rwr_personalization(valid_nodes, node_specificity, primekg_embs, primekg_map, context_nodes, idx2id)
    
    # [PERFORMANCE & MEMORY FIX] 
    # 只在這篇病歷事前跑一次 RWR，避免在迴圈內每個疾病都重複產生巨大的 Sparse 記憶體分配
    note_pr_dict = {}
    note_disease_vals = np.array([])
    if note_personalization:
        try:
            Q, is_dangling, nodelist, node_index_map, unweighted_degree = pr_comps
            N_nodes = len(nodelist)
            p = np.zeros(N_nodes)
            for n, val in note_personalization.items():
                if n in node_index_map:
                    p[node_index_map[n]] = val
                    
            p_sum = p.sum()
            p = p / p_sum if p_sum > 0 else np.repeat(1.0 / N_nodes, N_nodes)
            
            x = p.copy()
            alpha_pr = 1 - restart_rate  # 模擬 nx.pagerank alpha (設定 restart 機率)
            
            for _ in range(50):
                xlast = x
                x = alpha_pr * (x @ Q + np.sum(x[is_dangling]) * p) + (1 - alpha_pr) * p
                if np.abs(x - xlast).sum() < 1e-6:
                    break
                    
            note_pr_dict = {nodelist[j]: x.item(j) for j in range(N_nodes)}
            note_disease_vals = np.array([note_pr_dict.get(d, 0.0) for d in disease_nodes])
        except Exception as e:
            tqdm.write(f"Fast PR Error: {e}")

    results = []
    seen = set()
    target_mapping_details = []
    target_total_count = 0
    target_mapped_count = 0

    i = 0
    t_list = gen_tokens.tolist()
    skip_c = {'-', ' ', '.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '*', '•', '\n'}
    
    # 3. PARSE DISEASES & SCORE
    extracted_targets = []
    awaiting_bullet = True
    
    while i < len(t_list):
        t_str = gen_tokenizer.decode([t_list[i]])
        
        if '\n' in t_str:
            awaiting_bullet = True
            i += 1
            continue
            
        t_strip = t_str.strip()
        if not t_strip:
            i += 1
            continue
            
        if awaiting_bullet:
            awaiting_bullet = False
            if t_strip[0] in ['-', '*', '•', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                start_idx = i
                found_alpha = False
                
                # 尋找疾病名稱真正開始的那一個 Token (包含英文字母即算起點)
                for search_idx in range(i, min(i+7, len(t_list))):
                    c_str = gen_tokenizer.decode([t_list[search_idx]])
                    if any(c.isalpha() for c in c_str):
                        start_idx = search_idx
                        found_alpha = True
                        break
                        
                if found_alpha:
                    end_idx = start_idx
                    while end_idx < len(t_list) and '\n' not in gen_tokenizer.decode([t_list[end_idx]]):
                        end_idx += 1
                        
                    # 從結尾扣除純粹空白或沒字母的 Token，避免把 \n 或空白算進去
                    actual_end = end_idx - 1
                    while actual_end >= start_idx and not any(c.isalpha() for c in gen_tokenizer.decode([t_list[actual_end]])):
                        actual_end -= 1
                        
                    if actual_end >= start_idx:
                        orig_name = gen_tokenizer.decode(t_list[start_idx:actual_end+1]).replace('<|eot_id|>', '').strip()
                        orig_name = orig_name.lstrip('-*•1234567890. \\t')
                        
                        if orig_name.lower().startswith('note:') or orig_name.lower().startswith('note '):
                            break
                            
                        orig_name = re.sub(r'\(.*?\)|\[.*?\]', '', orig_name)
                        orig_name = re.sub(r'\(.*$|\[.*$', '', orig_name)
                        
                        # 處理模型過度解釋 (Explanations) 的後綴句：遇到以下停用詞就強制切斷字串
                        orig_name = re.split(r'(?i)\s+(cannot be|due to|secondary to|despite|likely|possible|probable|concerning for|rule out|suspected)', orig_name)[0]
                        orig_name = orig_name.strip(' "\'')
                        
                        if len(orig_name) > 2 and orig_name.lower() not in seen:
                            seen.add(orig_name.lower())
                            extracted_targets.append((orig_name, start_idx, actual_end))
                    i = end_idx
                    continue
        i += 1

    # [KG 局部 Min-Max Scaling 預處理]
    # 先把這篇病歷所有生成的疾病做完 Mapping 並拿到 RWR 機率，找出最大最小值
    target_mappings = []
    target_scores = []
    threshold_scores = []
    valid_targets = [] # 保存未被過濾掉的 targets
    for orig_name, start_idx, actual_end in extracted_targets:
        dis_nodes_mapped, mapping_detail = map_to_kg([orig_name], kg_embs, valid_idx, valid_names, nli_model, nli_tokenizer, entailment_id, return_details=True)
        
        # [修改] 針對 Generation Model 生成的疾病，若無 mapping 或 type 不符，將清空 mapping 結果變成未 mapped 狀態，但仍保留不丟棄，kg_only 的分數為 0
        is_mapped_valid = False
        if dis_nodes_mapped:
            mapped_type = idx2type.get(dis_nodes_mapped[0], "").lower()
            if "disease" in mapped_type or "effect" in mapped_type or "phenotype" in mapped_type:
                is_mapped_valid = True
                
        if not is_mapped_valid:
            dis_nodes_mapped = []

        valid_targets.append((orig_name, start_idx, actual_end))        
        target_total_count += 1
        if dis_nodes_mapped:
            target_mapped_count += 1
            if mapping_detail:
                target_mapping_details.extend(mapping_detail)
                
        t_score = 0.0
        t_score_thresh = 0.0
        if dis_nodes_mapped and note_pr_dict:
            n = dis_nodes_mapped[0]
            t_score = note_pr_dict.get(n, 0.0)
            t_score_thresh_base = t_score

            # 為了「比較用的門檻」，如果它是起點，也必須扣除 restart bias 以求公平
            if note_personalization and n in note_personalization:
                t_score_thresh_base -= restart_rate * note_personalization[n]
                t_score_thresh_base = max(0.0, t_score_thresh_base)

            # 原本生成的疾病保留原始 RWR 機率算最終分數，以反映它們身為已知線索的真實價值
            # 但需要扣除 degree 的影響: 避免被 degree 高的 hub 節點主導
            if n in node_index_map:
                deg = unweighted_degree[node_index_map[n]]
                deg_log = np.log10(deg + 10.0)
                t_score = t_score / deg_log
                t_score_thresh = t_score_thresh_base / deg_log

        target_mappings.append((dis_nodes_mapped, mapping_detail, t_score))
        target_scores.append(t_score)
        threshold_scores.append(t_score_thresh)
        
    valid_scores = [s for s in target_scores if s > 0.0]    
    local_min = min(valid_scores) if valid_scores else 0.0
    local_max = max(valid_scores) if valid_scores else 0.0

    valid_threshold_scores = [s for s in threshold_scores if s > 0.0]
    expansion_threshold = np.mean(valid_threshold_scores) if valid_threshold_scores else 0.0

    # 收集已生成的 mapped 節點 ID，用於擴充時排除
    already_generated_nodes = set()
    for mapped_list, _, _ in target_mappings:
        if mapped_list:
            already_generated_nodes.add(mapped_list[0])
            
    # === [Expansion Mechanism] ===
    expansion_targets = []
    if note_pr_dict:
        exp_candidates = []
        for n in disease_nodes:
            # 確保不會再挑到已經用 generation model 生成過的疾病實體
            if n in already_generated_nodes:
                continue
                
            if n in note_pr_dict:
                score = note_pr_dict[n]
                # 扣除起點自身的 restart bias (避免 self-contribution 造成資訊流出)
                if note_personalization and n in note_personalization:
                    score -= restart_rate * note_personalization[n]
                    score = max(0.0, score)  # 防止浮點數誤差造成負數
                
                # 扣除 degree 的影響: 避免被 degree 高的 hub 節點主導
                if n in node_index_map:
                    deg = unweighted_degree[node_index_map[n]]
                    score = score / np.log10(deg + 10.0)
                
                exp_candidates.append((n, score))
                
        # 1. 取得所有在 disease_nodes 中的 nodes 及其機率，由大到小排序
        sorted_probs = sorted(exp_candidates, key=lambda x: x[1], reverse=True)
        
        # 2. & 3. 找出累積機率前 1% 的 nodes (改成限制最多抓取前 top_k 個，避免被單一巨型 hub 節點吃滿而產生中斷)
        total_prob = sum(score for _, score in sorted_probs)
        cum_prob = 0.0
        exp_nodes = []
        for n, score in sorted_probs:
            exp_nodes.append((n, score))
            cum_prob += score
            if len(exp_nodes) >= top_k_kg:  # 限制最多擴充候選 top_k_kg 個高機率疾病，防止只找到一兩個
                break
                
        # 4. 過濾條件: (a) 在第一次generation model沒有生成到 (b) 分數 > expansion_threshold (平均值)
        for n, score in exp_nodes:
            if score <= expansion_threshold:
                continue
                
            orig_name = idx2name.get(n, "").strip()
            # 去除括號及其內部內容
            orig_name = re.sub(r'\s*\(.*?\)', '', orig_name).strip()
            
            if not orig_name or orig_name.lower() in seen:
                continue
                
            # 排列進擴充名單
            expansion_targets.append((orig_name, n, score))
            seen.add(orig_name.lower())

    for (orig_name, start_idx, actual_end), (dis_nodes_mapped, mapping_detail, t_score) in zip(tqdm(valid_targets, desc=f"Note {idx_for_logging}", leave=False), target_mappings):
        # --- A. PURE LOGITS & METRICS (Teacher Forcing) ---
        # --- A. PURE LOGITS & METRICS (Teacher Forcing) ---
        # 1. 為了確保純淨，我們把 "\n- " 當作 Prompt 的延伸，只對疾病名稱算分
        prefix_text = "\n-"
        disease_name = " " + orig_name.lstrip() # 確保疾病名稱前面有空白，幫助 tokenizer 正常切字
        
        # 2. 分別 Tokenize 以取得精確長度
        prefix_ids = gen_tokenizer.encode(prefix_text, add_special_tokens=False)
        disease_ids = gen_tokenizer.encode(disease_name, add_special_tokens=False)
        
        # 3. 組合完整輸入
        extended_prompt_ids = torch.cat([prompt_ids, torch.tensor([prefix_ids], device=device)], dim=1)
        tf_input = torch.cat([extended_prompt_ids, torch.tensor([disease_ids], device=device)], dim=1)
        
        disease_len = len(disease_ids)
        
        # 4. 執行 Forward Pass
        with torch.no_grad():
            out_tf = gen_model(input_ids=tf_input)
            # 我們只需要最後 disease_len 個預測結果
            # 讀完最後一個字預測「下一個字」的 logit 對我們沒用，所以要捨棄 (-1)
            # 我們要抓取從「倒數第 disease_len + 1 個字」讀完後，所輸出的 logit
            tf_logits = out_tf.logits[0, -(disease_len + 1) : -1, :].to(torch.float16)
        
        del out_tf, tf_input
        
        # 5. 計算分數
        sub_logits = tf_logits.float()
        sub_probs = F.softmax(sub_logits, dim=-1)
        
        # 目標 ID 就是疾病本身的 ID
        sub_ids = torch.tensor(disease_ids, device=device).unsqueeze(1)
        prob_list = sub_probs.gather(1, sub_ids).squeeze(1).tolist()
        entropy_list = (-torch.sum(sub_probs * torch.log(sub_probs + 1e-9), dim=-1)).tolist()
        
        del sub_logits, sub_probs, sub_ids
        
        uq_pure = np.exp(np.mean(np.log(np.array(prob_list) + 1e-9))) if prob_list else 0.0
        uq_seq_prob = float(np.prod(prob_list)) if prob_list else 0.0
        uq_neg_ent = -float(np.mean(entropy_list)) if entropy_list else 0.0
        uq_logprob_max = float(np.max(np.log(np.array(prob_list) + 1e-9))) if prob_list else 0.0
        uq_logprob_min = float(np.min(np.log(np.array(prob_list) + 1e-9))) if prob_list else 0.0
        
        # --- B. CCP SCORE (Keep original logic and original sequence position) ---
        uq_ccp = calculate_ccp(gen_model, gen_tokenizer, prompt_ids, gen_tokens[:start_idx], orig_name, logits[start_idx], top_k_ccp)
        
        # --- C. KG AUGMENTED (Local Min-Max Scaling) ---
        kg_p = 0.0
        if dis_nodes_mapped:
            if local_max > local_min:
                kg_p = float((t_score - local_min) / (local_max - local_min))
                kg_p = min(1.0, max(0.0, kg_p))
            elif local_max > 0.0:
                kg_p = 1.0  # 全部的 mapping 皆一樣高且不為 0
            
        # 若原始生成的 uq_pure 分數過低 (< 0.01)，則強行取消它的 KG 分數
        if uq_pure < 0.01:
            kg_p = 0.0
            
        uq_kg = (alpha * kg_p) + ((1 - alpha) * uq_pure)
        
        results.append({
            'disease': orig_name,
            'uq_pure': uq_pure,
            'uq_seq_prob': uq_seq_prob,
            'uq_neg_ent': uq_neg_ent,
            'uq_max_logprob': uq_logprob_max,
            'uq_min_logprob': uq_logprob_min,
            'uq_ccp': uq_ccp,
            'uq_kg': uq_kg,
            'kg_only': kg_p,
            'is_expanded': False,
            'tf_logits_expanded': 0.0,
            'tf_min_logprob_expanded': 0.0
        })
        
        # [記憶體保護] 每個疾病算完後，立刻強制清理 PyTorch 在計算 CCP 時殘留的 GPU Cache
        gc.collect()
        torch.cuda.empty_cache()
        if libc is not None:
            libc.malloc_trim(0)

    # --- Process Expansion Targets (Teacher Forcing) ---
    for orig_name, n_idx, t_score in tqdm(expansion_targets, desc=f"Note {idx_for_logging} Exp", leave=False):
        # Teacher forcing input: 附加換行與 diseases 名稱
        prefix_text = "\n- "
        disease_name = " " + orig_name.lstrip() 
        
        prefix_ids = gen_tokenizer.encode(prefix_text, add_special_tokens=False)
        disease_ids = gen_tokenizer.encode(disease_name, add_special_tokens=False)
        
        extended_prompt_ids = torch.cat([prompt_ids, torch.tensor([prefix_ids], device=device)], dim=1)
        tf_input = torch.cat([extended_prompt_ids, torch.tensor([disease_ids], device=device)], dim=1)
        
        disease_len = len(disease_ids)
        
        with torch.no_grad():
            out_tf = gen_model(input_ids=tf_input)
            # 使用精準的對齊切片
            tf_logits = out_tf.logits[0, -(disease_len + 1) : -1, :].to(torch.float16)
            
        del out_tf, tf_input
        
        sub_logits = tf_logits.float()
        sub_probs = F.softmax(sub_logits, dim=-1)
        
        # 目標 ID 就是疾病本身的 ID
        sub_ids = torch.tensor(disease_ids, device=device).unsqueeze(1)
        prob_list = sub_probs.gather(1, sub_ids).squeeze(1).tolist()
        entropy_list = (-torch.sum(sub_probs * torch.log(sub_probs + 1e-9), dim=-1)).tolist()
        
        del sub_logits, sub_probs, sub_ids
        
        uq_pure = np.exp(np.mean(np.log(np.array(prob_list) + 1e-9))) if prob_list else 0.0
        uq_seq_prob = float(np.prod(prob_list)) if prob_list else 0.0
        uq_neg_ent = -float(np.mean(entropy_list)) if entropy_list else 0.0
        uq_logprob_max = float(np.max(np.log(np.array(prob_list) + 1e-9))) if prob_list else 0.0
        uq_logprob_min = float(np.min(np.log(np.array(prob_list) + 1e-9))) if prob_list else 0.0
        
        # 根據參數決定 Teacher Forcing 的閾值過濾
        drop_expansion = False
        if tf_threshold_type == 'min_logprob' and uq_logprob_min < tf_threshold_val:
            drop_expansion = True
        elif tf_threshold_type == 'pure_logits' and uq_pure < tf_threshold_val:
            drop_expansion = True
            
        if drop_expansion:
            gc.collect()
            torch.cuda.empty_cache()
            if libc is not None:
                libc.malloc_trim(0)
            continue
            
        # 擴充疾病不跑 CCP (節省時間)
        uq_ccp = 0.0
        
        kg_p = 0.0
        if local_max > local_min:
            kg_p = float((t_score - local_min) / (local_max - local_min))
            kg_p = min(1.0, max(0.0, kg_p))  # 防止擴充疾病的 t_score 太高導致分數爆掉 (例如 2557)
        elif local_max > 0.0:
            kg_p = 1.0
            
        uq_kg = (alpha * kg_p) + ((1 - alpha) * uq_pure)
        
        results.append({
            'disease': orig_name,
            'uq_pure': 0.0,
            'uq_seq_prob': 0.0,
            'uq_neg_ent': -100.0,
            'uq_max_logprob': -100.0,
            'uq_min_logprob': -100.0,
            'uq_ccp': 0.0,
            'uq_kg': uq_kg,
            'kg_only': kg_p,
            'is_expanded': True,
            'tf_logits_expanded': uq_pure,
            'tf_min_logprob_expanded': uq_logprob_min
        })
        
        gc.collect()
        torch.cuda.empty_cache()
        if libc is not None:
            libc.malloc_trim(0)

    del logits, prompt_ids, gen_tokens
    torch.cuda.empty_cache()
    
    # 進行 RRF (倒數排名融合)
    if len(results) > 0:
        df_res = pd.DataFrame(results)
        k_rrf = 10  # 設定為 10
        
        # 建立用於排名的有效 pure 分數
        # 如果是 expanded，使用 tf_logits_expanded，否則使用 uq_pure
        effective_pure = np.where(df_res['is_expanded'], df_res['tf_logits_expanded'], df_res['uq_pure'])
        df_res['effective_pure'] = effective_pure
        
        # 分別排名 effective_pure 與 kg_only
        rank_pure = df_res['effective_pure'].rank(ascending=False, method='min')
        rank_kg = df_res['kg_only'].rank(ascending=False, method='min')
        
        rrf_pure = 1.0 / (k_rrf + rank_pure)
        rrf_kg = 1.0 / (k_rrf + rank_kg)
        
        # 如果 effective_pure < 0.01，或是沒有 mapping 到 KG (kg_only == 0.0)，抹除它從 KG 得到的 RRF 加分
        rrf_kg = np.where((df_res['effective_pure'] < 0.01) | (df_res['kg_only'] == 0.0), 0.0, rrf_kg)
        
        df_res['uq_rrf'] = rrf_pure + rrf_kg
        
        # 移除臨時欄位，確保不影響後續輸出
        df_res = df_res.drop(columns=['effective_pure'])
        
        results = df_res.to_dict('records')

    return results, {
        'context_ents': context_ents,
        'context_nodes': context_nodes,
        'mapping_details': mapping_details,
        'target_mapping_details': target_mapping_details,
        'target_mapped_count': target_mapped_count,
        'target_total_count': target_total_count,
        'rwr_stats': rwr_stats
    }

# --- Main Runner ---
def main():
    args = parse_args()
    base_suffix = f"{args.mode}" if args.dataset == "mimic" else f"{args.dataset}_{args.mode}"
    suffix = base_suffix
    if args.model == 'qwen':
        suffix += "_qwen"
    elif args.model == 'mistral':
        suffix += "_mistral"
    elif args.model == 'gemma':
        suffix += "_gemma"
        
    out_dir = f"results/{args.dataset}"
    if args.model == 'qwen':
        out_dir = f"results/{args.dataset}/qwen"
    elif args.model == 'mistral':
        out_dir = f"results/{args.dataset}/mistral"
    elif args.model == 'gemma':
        out_dir = f"results/{args.dataset}/gemma"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, DEFAULT_OUTPUT_CSV.replace('.csv', f'_{suffix}.csv'))
    
    if args.dataset == 'mimic':
        input_csv = 'data/MIMIC_notes_icd_01.csv'
        text_col = 'discharge_summary'
    elif args.dataset == 'pmc':
        input_csv = 'data/qwen_results_pmc.csv'
        text_col = 'structured_output'
    elif args.dataset == 'mtsamples':
        input_csv = 'data/qwen_results_mtsamples.csv'
        text_col = 'structured_output'

    # 讀取 Step 1 的結果
    extracted_jsonl = os.path.join(f"results/{args.dataset}", f"qwen_entities_{base_suffix}.jsonl")
    if not os.path.exists(extracted_jsonl):
        print(f"找不到 {extracted_jsonl}！請先執行 step1_extract_entities.py 產生對應模式的實體檔案。")
        return
        
    extracted_dict = {}
    with open(extracted_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                extracted_dict[data['index']] = data['entities']
    
    df = pd.read_csv(input_csv)
    start, end = args.start_index, args.end_index or len(df)
    
    proc_set = set()
    if os.path.exists(out_csv):
        proc_set = set(pd.read_csv(out_csv)['index'].unique())
        print(f"Resuming {out_csv}, found {len(proc_set)} notes.")
    else:
        pd.DataFrame(columns=['index','disease','uq_pure','uq_seq_prob','uq_neg_ent','uq_max_logprob','uq_min_logprob','uq_ccp','uq_kg','kg_only','is_expanded', 'tf_logits_expanded', 'tf_min_logprob_expanded', 'uq_rrf']).to_csv(out_csv, index=False)
        
    valid_nodes, node_specificity, idx2name, idx2type, idx2id, dis_nodes, primekg_embs, primekg_map, pr_comps = load_primekg()
    load_models(model_choice=args.model)
    v_idx, v_names, kg_embs = precompute_kg_embeddings(idx2name, idx2type)
    
    for _, row in tqdm(df.iloc[start:end].iterrows(), total=end-start):
        idx = int(row['index'] if 'index' in row else row.name + 1)
        if idx in proc_set: continue
        
        # 取得這篇病歷預先抓好的 Entities
        pre_extracted = extracted_dict.get(idx, [])
        
        note = str(row[text_col])
        
        if args.dataset in ['pmc', 'mtsamples']:
            sections_to_remove_simple = [
                'PAST MEDICAL HISTORY:', 
                'SOCIAL AND FAMILY HISTORY:', 
                'PHYSICAL EXAMINATION:', 
                'DISCHARGE DIAGNOSIS:'
            ]
            if args.mode == 'sliced':
                sections_to_remove_simple.append('HOSPITAL COURSE:')
                
            for sec in sections_to_remove_simple:
                pattern = r'(?im)^\s*' + re.escape(sec) + r'.*?(?=^\s*[A-Z\s/-]+:|\Z)'
                note = re.sub(pattern, '\n', note, flags=re.DOTALL)
        else:
            # 定義要移除的 sections 以及用於作為停止標記的可能標頭
            sections_to_remove = [
                r'(?:Past Medical History|PMH|Past Psychiatric History|Medical History)',
                r'(?:Social History|Social Hx)',
                r'(?:Family History|Family Hx)',
                r'(?:SOCIAL/FAMILY HX|Social/Family History)',
                r'(?:Discharge Diagnosis|Discharge Diagnoses|Primary Diagnosis|Primary Diagnoses|Final Diagnosis|Final Diagnoses|Secondary Diagnoses)',
                r'(?:Discharge Medications|Discharge Meds|Medications on Discharge|Medications at Discharge)',
                r'(?:Inactive Issues|Chronic Issues)',
                r'(?:Medications on Admission|Admission Medications|Admission Meds|Medications at Admission|Pre-admission Medications|Preadmission Medications)'
            ]
            stop_headers = [
                r'Admission Date', r'Discharge Date', r'Date of Birth', r'Sex', r'Service',
                r'Allergies', r'Attending', r'Chief Complaint', r'Major Surgical or Invasive Procedure',
                r'History of Present Illness', r'HPI', r'Past Medical History', r'PMH', 
                r'Social History', r'Family History', r'Physical Exam', r'PE', r'Admission Physical Exam', 
                r'Physical Examination', r'Pertinent Results', r'Labs on Admission', r'Labs on Discharge', 
                r'Imaging', r'Brief Hospital Course', r'Hospital Course', r'Medications on Admission',
                r'Admission Medications', r'Discharge Medications', r'Discharge Disposition', 
                r'Discharge Diagnosis', r'Discharge Diagnoses', r'Primary Diagnosis', r'Final Diagnosis',
                r'Discharge Condition', r'Discharge Instructions', r'Followup Instructions', 
                r'Review of Systems', r'ROS', r'Inactive Issues', r'Admission Diagnosis', 
                r'Discharge Status', r'Condition on Discharge', r'Discharge', 
                r'Active Issues', r'Transitional Issues', r'Code Status', r'Disposition'
            ]
            
            # 不要將 (?im) 放在 header_pattern 中，以免在串接時引發 "global flags not at the start" 的錯誤
            header_pattern = r'^\s*(?:' + '|'.join(stop_headers) + r')\s*:'
            for sec in sections_to_remove:
                pattern = r'(?im)^\s*' + sec + r'\s*:.*?(?=' + header_pattern + r'|\Z)'
                note = re.sub(pattern, '\n', note, flags=re.DOTALL)
    
            if args.mode == 'sliced':
                m = re.search(r'(?i)\n\s*(Physical Exam|Physical Examination|PE|Admission PE|Pertinent Results)\s*:', note)
                if m: note = note[:m.start()]
            
        try:
            res, kg_stats = combined_uq_pipeline(
                note, args.top_k_ccp, kg_embs, v_idx, v_names, valid_nodes, dis_nodes, args.alpha, 
                pre_extracted_entities=pre_extracted, node_specificity=node_specificity, primekg_embs=primekg_embs, 
                primekg_map=primekg_map, idx2id=idx2id, idx2type=idx2type, idx2name=idx2name, pr_comps=pr_comps, tf_threshold_type=args.tf_threshold_type, tf_threshold_val=args.tf_threshold_val, top_k_kg=args.top_k_kg, restart_rate=args.restart_rate, idx_for_logging=idx
            )
            
            # ========================
            # KG Mapping Stats Tracking
            # ========================
            context_ents = kg_stats['context_ents']
            context_nodes = kg_stats['context_nodes']
            mapping_details = kg_stats['mapping_details']
            
            qwen_mapping_rate = len(context_nodes) / len(context_ents) if len(context_ents) > 0 else 0.0
            mapped_entities_str = " | ".join(mapping_details) if mapping_details else "None"
            
            target_mapping_details = kg_stats['target_mapping_details']
            target_total = kg_stats['target_total_count']
            target_mapped = kg_stats['target_mapped_count']
            kg_mapping_rate = (target_mapped / target_total) if target_total > 0 else 0.0
            kg_mapped_entities_str = " | ".join(target_mapping_details) if target_mapping_details else "None"
            
            stats_rows = [{
                'index': idx,
                'qwen_mapping_rate': qwen_mapping_rate,
                'qwen_similarities': mapped_entities_str,
                'num_starting_nodes': len(context_nodes),
                'kg_mapping_rate': kg_mapping_rate,
                'kg_similarities': kg_mapped_entities_str
            }]
            
            out_stats_csv = os.path.join(out_dir, OPTIONAL_STATS_CSV.replace('.csv', f'_{suffix}.csv'))
            df_stats = pd.DataFrame(stats_rows)
            stats_header = not os.path.exists(out_stats_csv)
            df_stats.to_csv(out_stats_csv, mode='a', header=stats_header, index=False)
            # ========================
            
            # ========================
            # RWR Initial Probability Stats Tracking
            # ========================
            rwr_stats_rows = []
            for s in kg_stats['rwr_stats']:
                s_copy = dict(s)
                s_copy['note_index'] = idx
                s_copy['node_name'] = idx2name.get(s['node_index'], '')
                rwr_stats_rows.append(s_copy)
                
            if rwr_stats_rows:
                out_rwr_csv = os.path.join(out_dir, RWR_STATS_CSV.replace('.csv', f'_{suffix}.csv'))
                df_rwr = pd.DataFrame(rwr_stats_rows)
                df_rwr = df_rwr[['note_index', 'node_name', 'C', 'S', 'W', 'prob']]
                rwr_header = not os.path.exists(out_rwr_csv)
                df_rwr.to_csv(out_rwr_csv, mode='a', header=rwr_header, index=False)
            # ========================
            
            rows = []
            for r in res:
                r['index'] = idx
                rows.append(r)
            if not rows: rows.append({'index':idx, 'disease':None, 'uq_pure':0, 'uq_seq_prob':0, 'uq_neg_ent':0, 'uq_max_logprob':0, 'uq_min_logprob':0, 'uq_ccp':0, 'uq_kg':0, 'kg_only':0, 'is_expanded':False, 'tf_logits_expanded':0.0, 'tf_min_logprob_expanded':0.0, 'uq_rrf':0.0})
            
            pd.DataFrame(rows)[['index','disease','uq_pure','uq_seq_prob','uq_neg_ent','uq_max_logprob','uq_min_logprob','uq_ccp','uq_kg','kg_only','is_expanded','tf_logits_expanded', 'tf_min_logprob_expanded', 'uq_rrf']].to_csv(out_csv, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"Error Index {idx}: {e}")
            
        gc.collect()
        torch.cuda.empty_cache()
        if libc is not None:
            libc.malloc_trim(0)

if __name__ == "__main__":
    main()