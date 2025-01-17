import os
from dotenv import load_dotenv
load_dotenv()

# Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ollama_project"

from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Function to create the prompt template
def create_prompt(question: str) -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "you are a helpful assistant"),
            ("user", f"question:{question}")
        ]
    )
    return prompt

#streamlit frameworks
st.title("LangChain with LLAMA2")
input_text = st.text_input("What is the question in your mind?")

# Initialize the OllamaLLM
llm = OllamaLLM(model="gemma:2b")
output_parser = StrOutputParser()

# Create the prompt chain
chain = create_prompt(input_text) | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))

