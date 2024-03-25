from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import sys

accelerator = Accelerator()
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "lmsys/vicuna-13b-v1.5-16k"
tokenizer = AutoTokenizer.from_pretrained(model_name) # Target Model
model = AutoModelForCausalLM.from_pretrained(
    model_name, # Target Model
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16)
model.eval()
for param in model.parameters():
    param.require_grad = False
df = pd.read_csv(f'persuasive-paraphrase-prompting/data/ss_split_files/mixtral_ss_{sys.argv[1]}.csv')
df = df.dropna()
prompts_all = df['ss_prompts'].tolist()

def prepare_prompts(prompts, tokenizer, batch_size=10):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side="left"    
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                # padding='longest', 
                padding=True,
                truncation=False, 
                # pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

accelerator.wait_for_everyone()
with accelerator.split_between_processes(prompts_all) as prompts:
    responses = []

    prompt_batches=prepare_prompts(prompts_all, tokenizer, batch_size=10)

    for prompts_tokenized in prompt_batches:
        outputs = model.generate(**prompts_tokenized, max_new_tokens=2000)
        outputs =[tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs)]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        responses.extend(decoded_outputs)

df['mistral_reponse'] = responses
df.to_csv(f'persuasive-paraphrase-prompting/data/mistral_split_files/attack_results_{sys.argv[1]}.csv', index=False)
