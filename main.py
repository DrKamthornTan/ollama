import ollama
import streamlit as st
import torch
from translate import Translator

st.set_page_config(page_title='DHV Local LLM', layout='wide')
st.title("DHV AI Startup Medical Chatbot with Meditron")

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# init models
if "model" not in st.session_state:
    st.session_state["model"] = ""

models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.selectbox("Choose your model", models)

def model_res_generator():
    if torch.cuda.is_available():
        # Set the global PyTorch device to GPU
        device = torch.device("cuda")
        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        # Use CPU if no GPU available
        device = torch.device("cpu")

    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter prompt here.."):
    # Translate Thai input to English
    translator = Translator(to_lang="en", from_lang="th")
    translated_prompt = translator.translate(prompt)

    st.session_state["messages"].append({"role": "user", "content": translated_prompt})

    with st.chat_message("user"):
        st.markdown(translated_prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})