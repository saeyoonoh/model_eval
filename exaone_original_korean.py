import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
from tqdm import tqdm
import transformers
from datasets import load_dataset

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

TARGET_NUM = 3
HF_TOKEN='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'
EXAONE_STATE_DICT_PATH = torch.load('/pvc/home-syl-new/simple_model_state_dict.pth')

# transformer version check
assert str(transformers.__version__) == "4.41.0"

tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")
model = AutoModelForCausalLM.from_pretrained( "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", trust_remote_code=True, use_auth_token = HF_TOKEN)

model.eval()
DEVICE='cpu'
model.to(DEVICE)

idx = 0
total_logits = []
total_tokens = []
with torch.no_grad():
    pbar = tqdm(total=TARGET_NUM)
    while idx < TARGET_NUM:
        with open(f'./sci-news-sum-kr-50/data/0{idx+1}.json') as f:
            dataset = json.load(f)
        text = '. '.join(dataset['sentences'])

        # test vector of length 256.
        testvec = tokenizer(text, return_tensors="pt").input_ids.to(DEVICE)[:, :256]
        # generate for 256.
        prefill_logits = model(testvec).logits # 1, prefill(256), V
        generated = model.generate(testvec, return_dict_in_generate=True, max_length=512, output_scores=True, do_sample=False)
        decode_logits = torch.stack(generated.scores, dim=1) # 1, gen(256), V
        logits = torch.cat((prefill_logits, decode_logits), dim=1).squeeze() # 512, V
        tokens = generated.sequences.squeeze() # 512
        total_logits.append(logits.cpu())
        total_tokens.append(tokens.cpu())
        idx += 1
        pbar.update(1)
    pbar.close()

torch.save(total_logits, str(transformers.__version__)+f'_exaone_original_logit_{DEVICE}_kr.pt')
torch.save(total_tokens, str(transformers.__version__)+f'_exaone_original_token_{DEVICE}_kr.pt')
