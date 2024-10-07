from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
token = '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'
import json
import torch

with open("e_config.json", "r") as f:
    config_dict = json.load(f)
custom_config = LlamaConfig.from_dict(config_dict)


tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", use_auth_token=token)
model = AutoModelForCausalLM.from_config(custom_config)
import transformers

#model_exaone = AutoModelForCausalLM.from_pretrained( "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", torch_dtype=torch.bfloat16, trust_remote_code=True, use_auth_token = token )

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

print(model)
model.eval()



input_text = "Create long stroy: Once upon a time"
input_text = "Create long story: FuriosAI is"
#input_text = "Create long story: My name is"

if str(transformers.__version__) == "4.31.0":
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids#.to('cuda:0')
else:
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids#.to('cuda:0')


with torch.no_grad():
    outputs = model(input_ids, labels=input_ids)
    logits = outputs.logits
    generated = model.generate(input_ids, return_dict_in_generate=True, max_length=200, output_scores=True, do_sample=False)

torch.save(generated.scores, str(transformers.__version__)+"_tensor_tuple_exaone_3.1.pt")
torch.save(generated.sequences, str(transformers.__version__)+"_exaone_id_3.1.pt")