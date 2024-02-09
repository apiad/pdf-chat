# Chat with your PDF

> A simple streamlit app to chat with a PDF file.

## Running

- Get an API key from [mistral.ai](https://mistral.ai).
- Create a virtualenv.
- Install `requirements.txt`.
- Create a file `.streamlit/secrets.toml` with a `MISTRAL_API_KEY="<your-mistral-key>"`.
- Run `streamlit run app.py`.

## Basic architecture

The appplication is a single-script [streamlit](https://streamlit.io) app.
It uses streamlit's chat UI elements to simulate a conversation with a chatbot that
can answer queries about an uploaded PDF document.

### PDF processing and indexing

On the sidebar, the user can upload a PDF file that is processed with
the `pypdf` module to obtain a plain text representation.

Afterward, the text is split into chunks of 2048 characters with an overlap of 1024.
Each chunk is embedded using the `mistral-embed` model from [mistral.ai](https://mistral.ai)
through the Python library `mistralai`.
Embedded chunks are stored in a flat in-memory index using `faiss`, where they can be
retrieved by vector similarity.

### Question answering

The model `mistral-small` is employed to answer questions on the PDF content.

The user query is first embedded with the same `mistral-embed` model and queried against
the `faiss` index, where the closet matching chunk is extracted.

Afterward, a custom prompt is constructed that contains both the user query
and the retrieved chunk of text, and fed to the LLM. The response is streamed back
to the application.

### Chat management

The application relies on `st.chat_message` and `st.chat_input` as the main UI elements.

To stream text and simulate a typing behavior, instead of the classic `st.write`,
all text is displayed with `st.write_stream`, that receives an iterable of text chunks.

The text received from the LLM already comes in an iterable, so only a simple wraping
is necessary to obtain each text fragment.
However, the custom text that is sometimes displayed by the chatbot (like the hello message)
must be transformed into an iterable of text fragments with a small delay (`time.sleep(1/250)`)
to simulate the typing effect.

Since streamlit is by default stateless, all messages sent to the chatbot and its replies
are stored in the session state, and rewritten (not streamed) at the begining of each execution,
to keep the whole conversation in the screen.

All of this is performed by a custom function `add_message` that streams a message
the first time and stores it in the session state.

### Limitations

- Only the last message is actually sent to the LLM so even though the whole conversation is
displayed all the time, every query is independent. That is, the chatbot has no access to
the previous context. This is easy to fix by passing the last few messages on the call to `client.chat_stream(...)`.
- To save resources, documents with more than 100 chunks will error. This number can be changed in the source code.
- There is no attempt to sanitize the input or output, so the model can behave erratically and exhibit biased and impolite replies if prompted with such intention.
- Only one chunk is retrieved for each query so if the question requires a longer context the model will give an incorrect answer.
- All LLMs are subject to hallucinations so always double check your responses.

## Collaboration

Code is MIT. Fork, modify, and pull-request if you fancy :)
