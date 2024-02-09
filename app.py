import os
import time

import numpy as np
import streamlit as st
from faiss import IndexFlatL2
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from pypdf import PdfReader


st.set_page_config(page_title="Chat with your PDF", page_icon="📝", layout="wide")


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


@st.cache_resource
def get_client():
    api_key = os.environ["MISTRAL_API_KEY"]
    return MistralClient(api_key=api_key)


CLIENT: MistralClient = get_client()


PROMPT = """
An excerpt from a document is given below.

---------------------
{context}
---------------------

Given the document excerpt, answer the following query.
If the context does not provide enough information, decline to answer.
Do not output anything that can't be answered from the context.

Query: {query}
Answer:
"""


def reply(query: str):
    embedding = CLIENT.embeddings(model="mistral-embed", input=query).data[0].embedding
    embedding = np.array([embedding])

    _, indexes = index.search(embedding, k=2)
    context = [st.session_state.chunks[i] for i in indexes.tolist()[0]]

    messages = [
        ChatMessage(role="user", content=PROMPT.format(context=context, query=query))
    ]
    response = CLIENT.chat_stream(model="mistral-small", messages=messages)

    add_message(stream_response(response))


def build_index():
    st.session_state.messages = []

    pdf_file = st.session_state.pdf_file

    if not pdf_file:
        st.session_state.clear()
        return

    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n\n"

    st.session_state.text = text

    st.sidebar.info(
        f"The uploaded PDF has {len(reader.pages)} pages and {len(text)} characters."
    )

    chunk_size = 1024
    chunks = [text[i : i + 2 * chunk_size] for i in range(0, len(text), chunk_size)]

    st.sidebar.info(f"Indexing {len(chunks)} chunks.")
    progress = st.sidebar.progress(0)

    embeddings = []
    for i, chunk in enumerate(chunks):
        embeddings.append(
            CLIENT.embeddings(model="mistral-embed", input=chunk).data[0].embedding
        )
        progress.progress((i + 1) / len(chunks))

    embeddings = np.array(embeddings)

    dimension = embeddings.shape[1]
    index = IndexFlatL2(dimension)
    index.add(embeddings)

    st.session_state.index = index
    st.session_state.chunks = chunks


def stream_str(s, speed=250):
    for c in s:
        yield c
        time.sleep(1 / speed)


def stream_response(response):
    for r in response:
        yield r.choices[0].delta.content


if st.sidebar.button("🔴 Reset conversation"):
    st.session_state.messages = []


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["agent"]):
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
[browse the code](https://github.com/apiad/pdf-chat) in Github.""",
        store=False,
    )

    add_message("To begin, please upload your PDF file in the sidebar.", store=False)


pdf = st.sidebar.file_uploader(
    "Upload a PDF file", type="PDF", key="pdf_file", on_change=build_index
)

if not pdf:
    st.stop()


index: IndexFlatL2 = st.session_state.index
query = st.chat_input("Ask something about your PDF")


if not st.session_state.messages:
    reply("In one sentence, what is this document about?")
    add_message("Ready to answer your questions.")


if query:
    add_message(query, agent="human", stream=False, store=True)
    reply(query)
