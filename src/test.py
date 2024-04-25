import textwrap

import google.generativeai as genai

from IPython.display import display, Markdown
import pathlib

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Configure API key
GOOGLE_API_KEY = "AIzaSyDjkyqmEZK5UjNuHerdomNwxCAO5Ist4uo"  # Replace 'YOUR_API_KEY' with your actual API key
genai.configure(api_key=GOOGLE_API_KEY)

# List available models
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)


model = genai.GenerativeModel('gemini-pro')
response = model.generate_content("What is the meaning of life?")
print(response.text)
