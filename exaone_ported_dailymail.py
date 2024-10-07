import torch
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
import transformers
from datasets import load_dataset
from collections import OrderedDict
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"


TARGET_NUM = 3
HF_TOKEN='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'
EXAONE_STATE_DICT_PATH = torch.load('/pvc/home-syl-new/simple_model_state_dict.pth')

# transformer version check
assert str(transformers.__version__) == "4.43.3"

# Load the CNN/DailyMail dataset
dataset = load_dataset("abisee/cnn_dailymail", '1.0.0')

# Load Exaone Config
with open("exaone_config.json", "r") as f:
    config_dict = json.load(f)
custom_config = LlamaConfig.from_dict(config_dict)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", use_auth_token=HF_TOKEN)
model = AutoModelForCausalLM.from_config(custom_config)

# Map Exaone param to LlamaModel
mapping_func = {'transformer': 'model', '.h.': '.layers.', 'ln_1': 'input_layernorm', 'ln_2': 'post_attention_layernorm', 'ln_f':'norm', 'wte': 'embed_tokens', 'c_fc_0': 'gate_proj', 'c_fc_1': 'up_proj', 'c_proj': 'down_proj', 'rotary': 'rotary_emb', '.attn.attention.':  '.self_attn.', 'out_proj': 'o_proj'}

def replace_key_in_ordered_dict(d, mapping_func):
    # 먼저 기존 OrderedDict의 키를 리스트로 저장
    keys = list(d.keys())
    for old_key in keys:
        new_key = old_key
        for key in mapping_func:
            new_key = new_key.replace(key, mapping_func[key])
        # new_key가 기존에 없거나 old_key와 다를 경우에만 변경
        if new_key != old_key:
            d[new_key] = d.pop(old_key)
    return d

aa = torch.load('/pvc/home-syl-new/simple_model_state_dict.pth')
bb = replace_key_in_ordered_dict(aa, mapping_func)

model.load_state_dict(bb, strict=False)
model.eval()
DEVICE='cpu'
model.to(DEVICE)
import ipdb; ipdb.set_trace()

idx = 0
total_logits = []
total_tokens = []
with torch.no_grad():
    pbar = tqdm(total=TARGET_NUM)
    while idx < TARGET_NUM:
        # test vector of length 256.
        testvec = tokenizer(dataset['test'][idx]['article'], return_tensors="pt").input_ids.to(DEVICE)[:, :256]
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

torch.save(total_logits, str(transformers.__version__)+f'_exaone_ported_logit_{DEVICE}.pt')
torch.save(total_tokens, str(transformers.__version__)+f'_exaone_ported_token_{DEVICE}.pt')

