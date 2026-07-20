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
prompt="suggest me a name for my food company"

message_system={
    "role" : "system",
    "content" : "You are my Brand manager ans suggest me name , in one word"
}

message={
    "role":role,
    "content":prompt
}

messages=[message_system,message]


# temprature by defualt is 0,range[0-2]
response=client.chat.completions.create(model=model, messages=messages, temperature=0)
print(response)

print("-----------------------------------------------------------------------------")

answer=response.choices[0].message.content
print(answer)
