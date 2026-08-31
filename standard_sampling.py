import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import re
import gc
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig
)

# --- Configuration ---
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
NLI_MODEL_PATH = "pritamdeka/PubMedBERT-MNLI-MedNLI"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DEFAULT_INPUT_CSV = 'data/MIMIC_notes_icd_01.csv'
DEFAULT_OUTPUT_CSV = 'standard_sampling.csv'

# --- Device Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# --- Model Loading Functions ---

def load_models(model_name):
    print(f"Loading Generative Model: {model_name}...")

    is_gemma = 'gemma' in model_name.lower()

    # Gemma 3 4B + 4-bit 量化會產生 inf/nan logits，導致 sampling 時 CUDA assertion
    # Gemma 3 4B 夠小，直接用 bfloat16 載入即可（約 8GB VRAM）
    if is_gemma:
        gen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            token=HF_TOKEN,
            attn_implementation="eager"
        )
    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        gen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device,
            trust_remote_code=True,
            token=HF_TOKEN,
            attn_implementation="sdpa"
        )
    gen_model.eval()

    tok_kwargs = dict(token=HF_TOKEN, padding_side="left")
    # Gemma 3 tokenizer 內部已經處理 BOS，重複加上 add_bos_token=True 會導致 assertion
    if not is_gemma:
        tok_kwargs["add_bos_token"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Generative Model ({model_name}) Loaded.")

    print(f"Loading NLI model: {NLI_MODEL_PATH}...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_PATH)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_PATH).to(device)
    nli_model.eval()

    entailment_id = 1 
    if nli_model.config.label2id:
        entailment_id = nli_model.config.label2id.get('entailment', 1)
        
    print(f"NLI Model Loaded. Using Entailment ID: {entailment_id}")
    
    return gen_model, tokenizer, nli_model, nli_tokenizer, entailment_id

# --- Helper Functions ---

nli_cache = {}

def create_clinical_prompt(note_text):
    return [
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

def check_nli_entailment(nli_model, nli_tokenizer, entailment_id, text_center, text_candidate):
    """
    Bidirectional entailment check with caching.
    """
    tc_clean = text_center.lower().strip()
    cand_clean = text_candidate.lower().strip()
    
    if tc_clean == cand_clean:
        return True
        
    pair_key = tuple(sorted([tc_clean, cand_clean]))
    if pair_key in nli_cache:
        return nli_cache[pair_key]
        
    # Check A -> B
    premise_a = f"The patient is diagnosed with {text_center}."
    hypothesis_b = f"The patient is diagnosed with {text_candidate}."
    
    # Check B -> A
    premise_b = f"The patient is diagnosed with {text_candidate}."
    hypothesis_a = f"The patient is diagnosed with {text_center}."
    
    inputs = nli_tokenizer(
        [premise_a, premise_b], 
        [hypothesis_b, hypothesis_a], 
        return_tensors="pt", truncation=True, padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = nli_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    
    entail_prob_a_b = probs[0, entailment_id].item() # A -> B
    entail_prob_b_a = probs[1, entailment_id].item() # B -> A
    
    is_entailed = (entail_prob_a_b > 0.5 or entail_prob_b_a > 0.5)
    nli_cache[pair_key] = is_entailed
    
    return is_entailed

def parse_diseases_from_text(text):
    diseases = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        # 匹配開頭的清單符號 (如 "-", "*", "•", "1.", "12.", "1)")
        m = re.match(r'^[-*•]\s*(.+)$|^\d+[\.\)]\s*(.+)$', line)
        if m:
            d_name = m.group(1) if m.group(1) else m.group(2)
            
            # 強制移除所有括號及其內部文字 (Post-processing)
            d_name = re.sub(r'\(.*?\)|\[.*?\]', '', d_name)
            d_name = re.sub(r'\(.*$|\[.*$', '', d_name)
            d_name = d_name.strip(' "\'')
            
            if len(d_name) > 1:
                diseases.append(d_name)
    return diseases

def run_standard_sampling(gen_model, tokenizer, nli_model, nli_tokenizer, entailment_id, note_text, num_samples=100, batch_size=1):
    # --- Context Window Management ---
    MAX_INPUT_TOKENS = 7300
    
    tokens_scan = tokenizer(note_text, add_special_tokens=False).input_ids
    if len(tokens_scan) > MAX_INPUT_TOKENS:
        tokens_scan = tokens_scan[-MAX_INPUT_TOKENS:]
        note_text = tokenizer.decode(tokens_scan)

    prompt_msgs = create_clinical_prompt(note_text)
    prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    
    if inputs.input_ids.shape[1] > 8192:
        inputs.input_ids = inputs.input_ids[:, -8192:]
        inputs.attention_mask = inputs.attention_mask[:, -8192:]

    input_len = inputs.input_ids.shape[1]
    
    all_extracted_diseases = [] # List of (sample_id, disease_string)
    accumulated_samples = 0
    
    # Using tqdm for inner progress
    with tqdm(total=num_samples, desc="  Inner Sampling", leave=False) as pbar:
        while accumulated_samples < num_samples:
            curr_batch = min(batch_size, num_samples - accumulated_samples)
            
            batch_input_ids = inputs.input_ids.repeat(curr_batch, 1)
            batch_attention_mask = inputs.attention_mask.repeat(curr_batch, 1)
            
            with torch.no_grad():
                outputs = gen_model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            for i in range(curr_batch):
                gen_seq = outputs[i][input_len:]
                gen_text = tokenizer.decode(gen_seq, skip_special_tokens=True)
                diseases = parse_diseases_from_text(gen_text)
                
                current_sample_id = accumulated_samples + i
                for d in diseases:
                    all_extracted_diseases.append((current_sample_id, d))
            
            accumulated_samples += curr_batch
            pbar.update(curr_batch)
        
    # --- Clustering / Aggregation ---
    clusters = [] 
    
    for sample_id, d_candidate in all_extracted_diseases:
        matched = False
        for cluster in clusters:
            if d_candidate.lower() == cluster['name'].lower():
                cluster['sample_ids'].add(sample_id)
                cluster['variations'].add(d_candidate)
                matched = True
                break
            
            # Use passed NLI model/tokenizer
            if check_nli_entailment(nli_model, nli_tokenizer, entailment_id, cluster['name'], d_candidate):
                 cluster['sample_ids'].add(sample_id)
                 cluster['variations'].add(d_candidate)
                 matched = True
                 break
        
        if not matched:
            clusters.append({
                'name': d_candidate,
                'sample_ids': {sample_id},
                'variations': {d_candidate}
            })
            
    results = []
    for c in clusters:
        score = len(c['sample_ids']) / num_samples
        results.append({
            'disease': c['name'],
            'uq': score
        })
        
    return results

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run Standard Sampling UQ Pipeline")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of notes to process (inclusive)")
    parser.add_argument("--end_index", type=int, default=None, help="End index of notes to process (exclusive). If not set, runs to the end.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV, help="Path to input CSV file")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Path to output CSV file")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for LLM generation")
    parser.add_argument("--mode", type=str, choices=['full', 'sliced'], default='full', help="Mode of input note: 'full' to keep it as is, 'sliced' to cut off at Physical Exam.")
    parser.add_argument("--text_column", type=str, default="discharge_summary", help="The name of the column in the CSV that contains the medical notes.")
    parser.add_argument("--dataset", type=str, choices=['mimic', 'pmc', 'mtsamples'], default='mimic', help="Choose dataset to process")
    parser.add_argument("--model", type=str, choices=['llama', 'qwen', 'mistral', 'gemma'], default='llama', help="Choose model to run")
    
    args = parser.parse_args()

    if args.dataset == 'pmc':
        args.input_csv = 'data/qwen_results_pmc.csv'
        args.text_column = 'structured_output'
        if args.output_csv == DEFAULT_OUTPUT_CSV:
            args.output_csv = 'standard_sampling_pmc.csv'
    elif args.dataset == 'mtsamples':
        args.input_csv = 'data/qwen_results_mtsamples.csv'
        args.text_column = 'structured_output'
        if args.output_csv == DEFAULT_OUTPUT_CSV:
            args.output_csv = 'standard_sampling_mtsamples.csv'

    out_dir = f"results/{args.dataset}"
    if args.model == 'qwen':
        out_dir = f"results/{args.dataset}/qwen"
    elif args.model == 'mistral':
        out_dir = f"results/{args.dataset}/mistral"
    elif args.model == 'gemma':
        out_dir = f"results/{args.dataset}/gemma"
        
    os.makedirs(out_dir, exist_ok=True)
    
    suffix = f"_{args.mode}"
    if args.model == 'qwen':
        suffix = f"_qwen_{args.mode}"
    elif args.model == 'mistral':
        suffix = f"_mistral_{args.mode}"
    elif args.model == 'gemma':
        suffix = f"_gemma_{args.mode}"

    output_csv = args.output_csv.replace('.csv', f'{suffix}.csv') if args.output_csv.endswith('.csv') else f"{args.output_csv}{suffix}"
    output_csv = os.path.join(out_dir, os.path.basename(output_csv))

    # 1. Read Data
    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input file not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    print(f"Total notes in dataset: {len(df)}")

    # 2. Determine Range
    start_idx = args.start_index
    end_idx = args.end_index if args.end_index is not None else len(df)
    
    # Ensure bounds
    if start_idx < 0: start_idx = 0
    if end_idx > len(df): end_idx = len(df)
    
    print(f"Processing range: Index {start_idx} to {end_idx} (Total: {end_idx - start_idx} notes)")

    # 3. Check Resume Status (Optional logic depending on if we want to overwrite or append)
    # We will append to output_csv. If it doesn't exist, create it.
    processed_indices = set()
    if os.path.exists(output_csv):
        try:
            df_existing = pd.read_csv(output_csv, usecols=['index'])
            processed_indices = set(df_existing['index'].unique())
            print(f"Found {len(processed_indices)} already processed notes in {output_csv}.")
        except (pd.errors.EmptyDataError, ValueError):
            print("Output file exists but might be empty.")
    else:
        print(f"Creating new output file: {output_csv}")
        pd.DataFrame(columns=['index', 'disease', 'uq']).to_csv(output_csv, index=False)

    # 4. Load Models (Load only if there is work to do)
    # Filter indices to run
    indices_to_run = [i for i in range(start_idx, end_idx)]
    
    # Simple check if all in range are done
    remaining_indices = [i for i in indices_to_run if df.iloc[i].get('index', i) not in processed_indices]
    
    if not remaining_indices:
        print("All notes in the specified range have been processed. Exiting.")
        return

    # Load models now
    model_name_map = {
        'llama': "meta-llama/Llama-3.1-8B-Instruct",
        'qwen': "Qwen/Qwen2.5-7B-Instruct",
        'mistral': "mistralai/Mistral-7B-Instruct-v0.3",
        'gemma': "google/gemma-3-4b-it"
    }
    chosen_model = model_name_map[args.model]
    gen_model, tokenizer, nli_model, nli_tokenizer, entailment_id = load_models(model_name=chosen_model)

    # 5. Main Loop
    print("Starting processing loop...")
    
    # Iterate specifically over the requested range
    for idx in tqdm(range(start_idx, end_idx), desc="Processing Notes"):
        row = df.iloc[idx]
        current_index = row.get('index', idx)
        
        if current_index in processed_indices:
            # tqdm.write(f"Skipping Index {current_index} (Already processed)")
            continue
            
        try:
            note_text = str(row[args.text_column])
            
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
                    note_text = re.sub(pattern, '\n', note_text, flags=re.DOTALL)
                    
                if args.mode == 'sliced':
                    tqdm.write(f"\n{'='*20} Index {current_index} Sliced Input {'='*20}")
                    tqdm.write(note_text)
                    tqdm.write(f"{'='*60}\n")
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
                
                header_pattern = r'^\s*(?:' + '|'.join(stop_headers) + r')\s*:'
                for sec in sections_to_remove:
                    pattern = r'(?im)^\s*' + sec + r'\s*:.*?(?=' + header_pattern + r'|\Z)'
                    note_text = re.sub(pattern, '\n', note_text, flags=re.DOTALL)
                
                if args.mode == 'sliced':
                    pattern_sliced = r'(?i)\n\s*(Physical Exam|Physical Examination|PE|Admission PE|Pertinent Results)\s*:'
                    match = re.search(pattern_sliced, note_text)
                    if match:
                        note_text = note_text[:match.start()]
                    
                    # 印出切斷後實際餵給模型的 input
                    tqdm.write(f"\n{'='*20} Index {current_index} Sliced Input {'='*20}")
                    tqdm.write(note_text)
                    tqdm.write(f"{'='*60}\n")
            
            uq_results = run_standard_sampling(
                gen_model, tokenizer, nli_model, nli_tokenizer, entailment_id,
                note_text, num_samples=100, batch_size=args.batch_size
            )
            
            # Prepare output
            output_rows = []
            for res in uq_results:
                output_rows.append({
                    'index': current_index,
                    'disease': res['disease'],
                    'uq': res['uq']
                })
                
            if not output_rows:
                 output_rows.append({
                    'index': current_index,
                    'disease': None,
                    'uq': None
                })
                
            # Write to CSV immediately
            df_out = pd.DataFrame(output_rows)
            df_out = df_out[['index', 'disease', 'uq']]
            df_out.to_csv(output_csv, mode='a', header=False, index=False)
            
            tqdm.write(f"Index {current_index}: Found {len(output_rows)} distinct clusters.")
            
        except Exception as e:
            print(f"\n[Error] Index {current_index}: {e}")
            continue
        
        # GC every 5 notes
        if idx % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
