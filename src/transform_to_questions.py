# Import necessary modules
import json
import os
import re
import json5  # Ensure this is installed: pip install json5
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = ''

### VTTs to Q/A Pairs ###
# Load and extract text from the input JSONL file
input_file = 'E:/jworg/vtts/vtts.json'
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# If your file contains multiple entries, you can loop through them.
text_content = data[900]['text']  # Adjust the index if necessary
title_content = data[900]['title']  # Adjust the index if necessary
formatCode_content = data[900]['formatCode']  # Adjust the index if necessary
duration_content = data[900]['duration']/60  # Adjust the index if necessary

# Initialize the ChatGPT model with temperature 0.2
chat_model = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.2)

# Create a prompt to instruct the model to generate Q/A pairs
prompt_template = PromptTemplate(
    input_variables=["text", "title", "formatCode", "duration"],
    template="""
Por favor, genera una lista de preguntas y respuestas basadas en el siguiente texto. 
Las preguntas deben ser imaginativas y variadas, no genéricas. Deben poder entenderse sin ningún contexto previo sobre el texto. Debe tener suficiente información en la propia pregunta para que pueda ser respondida unequívocamente con el texto, conteniendo información específica que haga referencia a la publicación exacta. Deben ser también variopintas de un estilo humano, pero centradas en el texto. Debe haber tanto preguntas largas y detalladas, como preguntas cortas y medias. Una pregunta al menos debe mencionar el título.
Las respuestas deben ser por lo general extensas, precisas y extraídas únicamente de la información proporcionada, con detalles interesantes.

**Importante**:

- Solo proporciona el resultado en formato JSON válido, sin texto adicional, sin comentarios y sin bloques de código.
- Asegúrate de que cada objeto en la lista esté separado por comas.
- Asegúrate de que todos los corchetes y llaves de cierre están presentes.
- El JSON debe tener la siguiente estructura:

[
  {{
    "messages": [
      {{"role": "user", "content": "Pregunta 1"}},
      {{"role": "assistant", "content": "Respuesta 1"}}
    ]
  }},
  {{
    "messages": [
      {{"role": "user", "content": "Pregunta 2"}},
      {{"role": "assistant", "content": "Respuesta 2"}}
    ]
  }}
  // ... más pares de preguntas y respuestas
]

Tipo de formato:
{formatCode}

Título:
{title}

Duración:
{duration} minutos

Texto:
{text}

Q/A Pairs:
"""
)

# Format the prompt with the text content
prompt = prompt_template.format(text=text_content, title=title_content, formatCode=formatCode_content, duration=duration_content)

# Generate the response from the model
response = chat_model([HumanMessage(content=prompt)])

# Extract the generated text
generated_text = response.content

# Print the generated text for debugging
print("Generated Text:")
print(generated_text)

# Remove code fences and extract JSON content
generated_text = generated_text.strip()
if generated_text.startswith("```json"):
    generated_text = generated_text[len("```json"):].strip()
elif generated_text.startswith("```"):
    generated_text = generated_text[len("```"):].strip()
if generated_text.endswith("```"):
    generated_text = generated_text[:-len("```")].strip()

# Fix common JSON errors
def fix_json_errors(text):
    # Remove any trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    # Add missing commas between objects
    text = re.sub(r'}\s*{', '},\n{', text)
    # Ensure the JSON array is properly enclosed
    if not text.startswith('['):
        text = '[' + text
    if not text.endswith(']'):
        text = text + ']'
    return text

json_content = fix_json_errors(generated_text)

# Try to parse the JSON content
try:
    output_data = json5.loads(json_content)
except ValueError as e:
    print(f"Error parsing JSON: {e}")
    # Save the generated text to a file so you don't lose data
    with open('E:/jworg/generated_output.txt', 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print("Generated text saved to E:/jworg/generated_output.txt for manual inspection.")
    output_data = []

# Write the output data to a new JSONL file
output_file = 'E:/jworg/qa_data.jsonl'
with open(output_file, 'a', encoding='utf-8') as f:
    if isinstance(output_data, list) and output_data:
        for entry in output_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write('\n')
        print(f"Generated {len(output_data)} Q/A pairs and saved to {output_file}.")
    elif isinstance(output_data, dict):
        json.dump(output_data, f, ensure_ascii=False)
        f.write('\n')
        print(f"Generated 1 Q/A pair and saved to {output_file}.")
    else:
        print("No valid JSON data to write.")