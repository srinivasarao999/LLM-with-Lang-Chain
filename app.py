import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import deepspeed
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the DeepSpeed model and tokenizer
MODEL_NAME = os.getenv("DEEPSPEED_MODEL_NAME", "EleutherAI/gpt-neo-2.7B")  # Change this to your model name

# Initialize the tokenizer and model for DeepSpeed
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Load the model into DeepSpeed
deepspeed.init_distributed()

# For DeepSpeed optimization
model = deepspeed.init_inference(model, dtype=torch.float16)  # Optimizing with fp16, can adjust for more resources

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With DeepSpeed"

## Prompt Template
def generate_response(question, max_tokens, temperature):
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Run inference using DeepSpeed
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_length=max_tokens,
            num_beams=5,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            length_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer


## Streamlit UI
st.title("Enhanced Q&A Chatbot With DeepSpeed")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your DeepSpeed API Key:", type="password")

## Select the model (DeepSpeed model is already chosen above)
engine = st.sidebar.selectbox("Select Open AI model", ["gpt-neo", "gpt-j", "gpt-2"])

## Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input and api_key:
    response = generate_response(user_input, max_tokens, temperature)
    st.write(response)

elif user_input:
    st.warning("Please enter the DeepSpeed API Key in the sidebar")
else:
    st.write("Please provide user input")
