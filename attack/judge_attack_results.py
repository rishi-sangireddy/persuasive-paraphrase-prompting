from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf") # Judge Model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-13b-chat-hf", # Judge Model
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16)

df = pd.read_csv('data/attack_results_ll7b.csv')
prompts_all = df['attack_reponse'].tolist()[:10]

## TODO: Get Judge Prompt to score the attack response and generate the prompts file for it

def prepare_prompts(prompts, tokenizer, batch_size=10):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.padding_side="left"     
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                padding='longest', 
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

accelerator.wait_for_everyone()
with accelerator.split_between_processes(prompts_all) as prompts:
    responses = []
    
    prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=10)
    
    for prompts_tokenized in prompt_batches:
        outputs = model.generate(**prompts_tokenized, max_new_tokens=256)
        outputs =[tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs)]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        responses.extend(decoded_outputs)

df['attack_reponse'] = responses
df.to_csv('data/Judge_results_ll13b.csv', index=False)
