import sys
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import pandas as pd
import time
api_key = "<API-KEY>"
model = "open-mixtral-8x7b"

df = pd.read_csv(f'persuasive-paraphrase-prompting/data/pap_split_files/pap_{sys.argv[1]}.csv')

client = MistralClient(api_key=api_key)
def generate_pap(prompt):
    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    # No streaming
    chat_response = client.chat(
        model=model,
        messages=messages,
        temperature=0.7
    )
    return chat_response.choices[0].message.content
i=1
for index, row in df.iterrows():
    if sys.argv[1]=='8' and i<508:
        i+=1
        continue
    # output = []
    # for _ in range(10):
    #     output.append(generate_pap(row['pap_prompt']))
    row['ss_prompts'] = generate_pap(row['pap_prompt'])
    ndf = pd.DataFrame([row])
    if i==1:
        ndf.to_csv(f'persuasive-paraphrase-prompting/data/ss_split_files/mixtral_ss_{sys.argv[1]}.csv', index=False)
    else:
        ndf.to_csv(f'persuasive-paraphrase-prompting/data/ss_split_files/mixtral_ss_{sys.argv[1]}.csv', mode = 'a', header=False, index=False)
    print(i)
    i+=1
    time.sleep(0.25)


