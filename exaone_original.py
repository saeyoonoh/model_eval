import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers

token = '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'

tokenizer = AutoTokenizer.from_pretrained("LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct")

model = AutoModelForCausalLM.from_pretrained( "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct", trust_remote_code=True, use_auth_token = token)


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