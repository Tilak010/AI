import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha h dost")
 
client=Groq(api_key=my_api_key)

model="llama-3.3-70b-versatile"
role="user"
# 3 prompts
prompt1="Hi!"
prompt2="Explain F1 basics in detail under 100 words"
prompt3="Write a 1000 words essay on Machine learning"

prompts = [prompt1,prompt2,prompt3]
for prompt in prompts:
    message={
    "role":role,
    "content":prompt
     }
    messages=[message]
    response=client.chat.completions.create(model=model, messages=messages, max_tokens=500)
    usage=response.usage
    print(f"promt: {prompt} -->your tokens: {usage.prompt_tokens} completion_tokens: {usage.completion_tokens} total tokens:{usage.total_tokens} Finish Reason: {response.choices[0].finish_reason}")



# message_system={
#     "role" : "system",
#     "content" : "You are my Brand manager ans suggest me name , in one word"
# }

# message={
#     "role":role,
#     "content":prompt
# }

# messages=[message_system,message]


# # temprature by defualt is 0,range[0-2]
# response=client.chat.completions.create(model=model, messages=messages, temperature=0)
# print(response)

# print("-----------------------------------------------------------------------------")

# answer=response.choices[0].message.content
# print(answer)
