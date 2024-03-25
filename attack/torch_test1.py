# Load model directly
from accelerate import Accelerator
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sys_prompts import get_topic_sys_prompt
accelerator = Accelerator()
pap_model_name = "lmsys/vicuna-13b-v1.5-16k"
topic_judge_model = "meta-llama/Llama-2-13b-chat-hf"
def load_model(model_name, device_index):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name,
        device_map={"": device_index},
        torch_dtype=torch.bfloat16)
    model.eval()
    for param in model.parameters():
        param.require_grad=False
    return tokenizer, model

def generate_response(model, tokenizer, prompt):
    tokens = tokenizer(prompt, return_tensors = "pt").to('cuda:0')
    output = model.generate(**tokens, max_new_tokens=1024, do_sample=True, temperature=0.7, num_beams=4)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return decoded_output.replace(prompt, "").replace('"', '').replace("Answer:", "").replace("youranswer1", "").replace("=", "").replace("\n", "").strip()

def generate_on_topic_response(model, tokenizer, sys_prompt, prompt):
    tokens = tokenizer("<<SYS>>"+sys_prompt+"<</SYS>>\n"+prompt, return_tensors = "pt").to('cuda:2')
    # sys_tokens = tokenizer(sys_prompt, return_tensors= 'pt').to('cuda:2')
    output = model.generate(**tokens, max_new_tokens=1024)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return "yes" in decoded_output.lower()

df = pd.read_csv('persuasive-paraphrase-prompting/data/pap_prompts.csv')
new_prompts= df['pap_prompt'].tolist()
queries = df['queries'].tolist()
# judge_topic_system_prompt = get_topic_sys_prompt()
pap = []
tokenizer, model = load_model(pap_model_name, 0)
judge_tokenizer, judge_model = load_model(topic_judge_model, 2)
# for i in range(0, 2000, 10):
#     prompts_all = new_prompts[i:i+10]
accelerator.wait_for_everyone()
with accelerator.split_between_processes(new_prompts) as prompts:
    responses = []
    for query, prompt in zip(queries, prompts):
        outputs = []
        for _ in range(10):
            res = generate_response(model, tokenizer, prompt)
            on_topic = generate_on_topic_response(judge_model, judge_tokenizer, get_topic_sys_prompt(query), res)
            if on_topic:
                break
            outputs.append(res)
        done = False
        if res=="":
            for out in outputs:
                if out!="":
                    responses.append(out)
                    done = True
                    break
        else:  
            responses.append(res)
        if not done:
            responses.append(res)
pap.extend(responses)

print(responses)
# df['ss_prompt'] = pap
# df.to_csv("ss_prompt_v13b-16k.csv", index=False)

