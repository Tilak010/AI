import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel

load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha h dost")

client = Groq(api_key=my_api_key)

# ------------------- Pydantic Model -------------------
class Ticket(BaseModel):
    name: str
    email: str
    issue: str

# Generate schema AFTER the class is created
schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema
and return ONLY valid JSON.

Schema:
{schema}
"""

# -------------------------------------------------------

text = """
Hello my name is Tilak. I have a PS5 which is not working that I bought from your shop.
My email is shrivastavatilak@gmail.com.
"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": text
    }
]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    temperature=0,
    response_format=response_format
)

# answer=print(response.choices[0].message.content)
# print (answer)

#isko padhna kese h
import json

answer = response.choices[0].message.content

# print(answer)

data = json.loads(answer)

ticket = Ticket(**data)

print(ticket.name)
print(ticket.email)
print(ticket.issue)