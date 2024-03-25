from accelerate import Accelerator
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch
import tensor_parallel as tp


# def get_gpu_memory(max_gpus=None):
#     """Get available memory for each GPU."""
#     import torch

#     gpu_memory = []
#     num_gpus = (
#         torch.cuda.device_count()
#         if max_gpus is None
#         else min(max_gpus, torch.cuda.device_count())
#     )

#     for gpu_id in range(num_gpus):
#         with torch.cuda.device(gpu_id):
#             device = torch.cuda.current_device()
#             gpu_properties = torch.cuda.get_device_properties(device)
#             total_memory = gpu_properties.total_memory / (1024**3)
#             allocated_memory = torch.cuda.memory_allocated() / (1024**3)
#             available_memory = total_memory - allocated_memory
#             gpu_memory.append(available_memory)
#     return gpu_memory

# available_gpu_memory = get_gpu_memory()

accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-33b-v1.3")
model = AutoModelForCausalLM.from_pretrained(
    "lmsys/vicuna-33b-v1.3",
    # device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16)
model = tp.tensor_parallel(model, [f"cuda:{i}" for i in range(torch.cuda.device_count())])


# device_map = accelerate.infer_auto_device_map(
#     model,
#     max_memory={i: str(int(available_gpu_memory[i] * 0.85)) + "GiB" for i in range(torch.cuda.device_count())},
#     no_split_module_classes=["LlamaDecoderLayer"],
# )
# model = accelerate.dispatch_model(
#     model, device_map=device_map, offload_buffers=True
# )

df = pd.read_csv('persuasive-paraphrase-prompting/data/new_prompts.csv')
prompts_all = df['prompt'].tolist()[:10]

def prepare_prompts(prompts, tokenizer, batch_size=2):
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

df['response'] = responses
df.to_csv('persuasive_prompts_v33b.csv', index=False)
