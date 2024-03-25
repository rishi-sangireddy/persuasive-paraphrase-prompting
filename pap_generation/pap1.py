from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import json
import torch

accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.5")
model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-13b-v1.5",
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16)

df = pd.read_csv('prompts.csv')
prompts_all = df['prompt'].tolist()

accelerator.wait_for_everyone()
i=1
with accelerator.split_between_processes(prompts_all) as prompts:
    responses = []
    for prompt in prompts:

        tokens = tokenizer(prompt, return_tensors = "pt").to('cuda')
        output = model.generate(**tokens, max_new_tokens=256)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        responses.append(decoded_output.replace(prompt, ""))
        print(i)
        i+=1
# pd.DataFrame(responses).to_csv('persuasive_prompts_10.csv', index=False)
df['response'] = responses
df.to_csv('persuasive_prompts.csv', index=False)