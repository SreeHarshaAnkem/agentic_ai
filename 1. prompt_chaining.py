'''This is the main module for prompt chaining functionality.'''
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
import json
from dotenv import load_dotenv
load_dotenv()

llm = init_chat_model("gpt-5-nano-2025-08-07", model_provider="openai")

# Step 1: Extract technical specifications from product description
prompt_extract = ChatPromptTemplate.from_template("Extract the technical specification from the following product description: {text_input}")

# Step 2: Transform technical specifications into a structured format
prompt_transform = ChatPromptTemplate.from_template("Transform the following technical specifications into a structured JSON format with cpu, memory, and storage as keys: {specifications}")

# Step 3: Build the chain using LCEL
chain = prompt_extract | llm | StrOutputParser() | prompt_transform | llm | JsonOutputParser()

# Step 4: Run the chain
input_text = "The new smartphone features a 2.8 GHz octa-core processor, 8GB RAM, and 256GB internal storage."
output = chain.invoke({"text_input": input_text})

# Step 5: Print the final structured output
print("Final Structured Output:", output)

# Step 6: Parse the JSON output
print("Type of output:", type(output))

