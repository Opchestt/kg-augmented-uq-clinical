import os
import re
import gc
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

HF_TOKEN = os.environ.get("HF_TOKEN", "")
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: Extract Entities using Qwen 72B")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=['full', 'sliced'], default='full')
    parser.add_argument("--dataset", type=str, choices=['mimic', 'pmc', 'mtsamples'], default='mimic', help="Choose dataset to process")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV (overrides dataset default)")
    return parser.parse_args()

def main():
    args = parse_args()
    suffix = f"{args.mode}" if args.dataset == "mimic" else f"{args.dataset}_{args.mode}"
    out_dir = f"results/{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    out_jsonl = os.path.join(out_dir, f"qwen_entities_{suffix}.jsonl")
    
    if args.dataset == 'mimic':
        input_csv = args.input_csv or 'data/MIMIC_notes_icd_01.csv'
        text_col = 'discharge_summary'
    elif args.dataset == 'pmc':
        input_csv = args.input_csv or 'data/qwen_results_pmc.csv'
        text_col = 'structured_output'
    elif args.dataset == 'mtsamples':
        input_csv = args.input_csv or 'data/qwen_results_mtsamples.csv'
        text_col = 'structured_output'

    proc_set = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, 'r') as f:
            for line in f:
                if line.strip():
                    proc_set.add(json.loads(line)['index'])
        print(f"Resuming {out_jsonl}, found {len(proc_set)} completed notes.")

    print("Loading Extractor Model (Qwen2.5-72B-Instruct-AWQ)...")
    ext_model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
    
    # 這裡使用 device_map="auto"，將大型模型均勻分布到可用 GPU 上
    extract_model = AutoModelForCausalLM.from_pretrained(
        ext_model_name, device_map="cuda", torch_dtype="auto", trust_remote_code=True, token=HF_TOKEN
    )
    extract_model.eval()
    extract_tokenizer = AutoTokenizer.from_pretrained(ext_model_name, token=HF_TOKEN)

    df = pd.read_csv(input_csv)
    start, end = args.start_index, args.end_index or len(df)
    
    for _, row in tqdm(df.iloc[start:end].iterrows(), total=end-start):
        idx = int(row['index'] if 'index' in row else row.name + 1)
        if idx in proc_set: continue
        
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
                m = re.search(r'(?i)\n\s*(Physical Exam|Physical Examination|PE|Admission PE)\s*:', note)
                if m: note = note[:m.start()]

        messages = [
            {"role": "system", "content": "You are an expert clinical knowledge extractor. Extract entities that match standard Medical Ontologies."},
            {"role": "user", "content": f"Extract ONLY a valid JSON array of strings containing documented diseases, past medical history, and primary symptoms from the text.\n\nText:\n{note}\n\nOutput:"}
        ]
        
        inputs = extract_tokenizer(extract_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(device)
        
        # 嚴格限制 Context 避免 OOM
        if inputs.input_ids.shape[1] > 7000:
            inputs.input_ids = inputs.input_ids[:, -7000:]
            inputs.attention_mask = inputs.attention_mask[:, -7000:]
            
        with torch.no_grad():
            out = extract_model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                temperature=None, 
                top_p=None, 
                top_k=None, 
                pad_token_id=extract_tokenizer.eos_token_id
            )
            
        raw = extract_tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        del inputs, out
        
        clean = re.sub(r'```json|```', '', raw).strip()
        entities = []
        try:
            arr = json.loads(clean)
            if isinstance(arr, list):
                entities = [re.sub(r'\(.*?\)|\[.*?\]', '', str(x)).strip() for x in arr]
        except:
            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                try: entities = [re.sub(r'\(.*?\)|\[.*?\]', '', str(x)).strip() for x in json.loads(match.group())]
                except: pass
                
        # 以 JSONL 格式增量寫入
        with open(out_jsonl, 'a') as f:
            f.write(json.dumps({'index': idx, 'entities': entities}) + '\n')
            
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()