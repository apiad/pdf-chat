import streamlit as st
import os
from pypdf import PdfReader
from faiss import IndexFlatL2
from mistralai.client import MistralClient
import time

st.set_page_config(page_title="Chat with your PDF", page_icon="📝")


if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Reset conversation"):
    st.session_state.messages = []


def add_message(msg, agent="ai", stream=True, store=True):
    if stream and isinstance(msg, str):
        msg = stream_str(msg)

    with st.chat_message(agent):
        if stream:
            output = st.write_stream(msg)
        else:
            output = msg
            st.write(msg)

    if store:
        st.session_state.messages.append(dict(agent=agent, content=output))


def stream_str(s, speed=250):
    for c in s:
        yield c
        time.sleep(1 / speed)


for message in st.session_state.messages:
    with st.chat_message(message['agent']):
        st.write(message["content"])


if not "text" in st.session_state:
    add_message(
        """
This is a simple demonstration of how to use a large language model
and a vector database to implement a bare-bones chat-with-your-pdf
application.

This appplication uses [Mistral](mistral.ai) as language model, so to
deploy it you will need a corresponding API key.

Read the [documentation](https://github.com/apiad/pdf-chat/blob/main/README.md) or
[browse the code](https://github.com/apiad/pdf-chat) in Github.""", store=False
    )

    add_message("To begin, please upload your PDF file in the sidebar.", store=False)


@st.cache_resource
def get_client():
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)


def upload_pdf():
    pdf_file = st.session_state.pdf_file

    if not pdf_file:
        st.session_state.pop("text")
        return

    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n\n"

    st.session_state.text = text

    add_message(f"The uploaded PDF has {len(reader.pages)} pages and {len(text)} characters. I will index it now.", store=False)

    client = get_client()
    index = IndexFlatL2()


st.sidebar.file_uploader("Upload a PDF file", type="PDF", key="pdf_file", on_change=upload_pdf)
