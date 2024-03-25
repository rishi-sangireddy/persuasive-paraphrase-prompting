from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch

accelerator = Accelerator()
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
# model_name = "lmsys/vicuna-13b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16)

df = pd.read_csv('persuasive-paraphrase-prompting/data/new_prompts.csv')
prompts_all = df['prompt'].tolist()

def prepare_prompts(prompts, tokenizer, batch_size=10):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side="left"  
    for prompt_batch in batches:
        batches_tok.append(
            tokenizer(
                prompt_batch, 
                return_tensors="pt", 
                # padding='longest', 
                padding=True,
                truncation=False, 
                pad_to_multiple_of=8,
                add_special_tokens=False).to("cuda") 
            )
    tokenizer.padding_side="right"
    return batches_tok

accelerator.wait_for_everyone()
with accelerator.split_between_processes(prompts_all) as prompts:
    responses = []
    
    prompt_batches=prepare_prompts(prompts, tokenizer, batch_size=5)
    
    for prompts_tokenized in prompt_batches:
        outputs = model.generate(
            **prompts_tokenized,
            do_sample=True,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=1,
            repetition_penalty = 1.0)
        outputs =[tok_out[len(tok_in):] for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs)]
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        responses.extend(decoded_outputs)

df['response'] = responses
df.to_csv('persuasive-paraphrase-prompting/data/persuasive_new_prompts_m7b_len.csv', index=False)
