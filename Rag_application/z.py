from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os
key=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
    task="text-generation",
    max_new_tokens=100,
)

print(llm.invoke("Who are you?"))

from huggingface_hub import InferenceClient

client = InferenceClient(token=key)

print(client)