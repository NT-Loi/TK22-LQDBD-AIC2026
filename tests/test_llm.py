import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

PROJECT_ID = os.environ.get("PROJECT_ID", "")
LOCATION = os.environ.get("VERTEX_LOCATION", "global")
model = os.environ.get("MODEL_ID")

client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

response = client.models.generate_content(
    model=model,
    contents=[
        "What is pikachu?",
    ],
    config=types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=True
        )
    ),
)
print(response.text, end="")