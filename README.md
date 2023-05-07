# [pdf-GPT: Load and Ask](https://virajsabhaya23-load-n-ask-app-syntut.streamlit.app/)

This is a Streamlit app that uses OpenAI's API to answer questions about your PDF file. 

<img width="675" alt="Screenshot 2023-05-07 at 15 19 31" src="https://user-images.githubusercontent.com/77448246/236704768-045a8a20-b1c2-47b6-8dd0-ddecf1d48fb7.png">

The app consists of the following main components:

- **PDF file uploader**: Upload your PDF file here.
- **OpenAI API key input**: Enter your OpenAI API key to use the OpenAI API for answering questions.
- **Ask a question**: Type your question here and the app will search for the answer in the uploaded PDF using the OpenAI API.
- **Past questions and answers**: This section displays the previously asked questions and their corresponding answers.

## Requirements

The following packages are required to run this app:

- streamlit
- streamlit-chat
- PyPDF2
- langchain
- faiss
- openai
- tiktoken
- requests

To install these packages, you can run:

```
pip install streamlit PyPDF2 langchain faiss openai
```

## How to use

To use this app, follow these steps:

1. Upload your PDF file.
2. Enter your OpenAI API key.
3. Ask your question about the PDF file.

The app will then search for the answer to your question in the uploaded PDF file using the OpenAI API. The answer will be displayed in the "Past questions and answers" section.

## About the author

This app was created by [Viraj Sabhaya](https://www.linkedin.com/in/vsabhaya23/). You can find the code for this app on [GitHub](https://github.com/virajsabhaya23/load_N_ask).
