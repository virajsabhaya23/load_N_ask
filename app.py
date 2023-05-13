from openai.error import OpenAIError
import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import os

# custom imports
from readPDF import read_pdf
# from utils import parse_docx, parse_text
from splitText import split_text


def create_embeddings(OPENAI_API_KEY, chunks):
    """
        Creates embeddings for each chunk
        :param OPENAI_API_KEY: OpenAI API key
        :param chunks: list of chunks
        :return: list of embeddings
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    retireved_info_knowledge = FAISS.from_texts(chunks, embeddings)
    return retireved_info_knowledge


def init_session_state():
    """
        Initializes session state
    """
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []


def ask_question(user_question, chunks, OPENAI_API_KEY):
    """
        Asks a question to the model
        :param user_question: question asked by user
        :return: answer to the question
    """
    # creating embeddings
    knowledge = create_embeddings(OPENAI_API_KEY, chunks)
    docs = knowledge.similarity_search(user_question)
    # st.write(docs)
    llms = OpenAI(openai_api_key=OPENAI_API_KEY)
    qa_chain = load_qa_chain(llms, chain_type="stuff")
    response = qa_chain.run(input_documents=docs, question=user_question)
    # st.write(response)
    return response


def uiSidebarInfo():
    """
        Displays information in sidebar
    """
    st.markdown("> version 1.0.2")
    # about this app and instructions CONTAINER
    with st.container():
        st.write("## ℹ️ About this app and instructions")
        with st.expander("Details ...", expanded=True):
            st.markdown(
                "This app uses [OpenAI](https://beta.openai.com/docs/models/overview)'s API to answer questions about your PDF file. \nYou can find the source code on [GitHub](https://github.com/virajsabhaya23/load_N_ask)."
            )
            st.markdown("1. Enter OpenAI API key.\n 2. Upload your PDF file. \n 3. Ask your question.")
    st.write("> made by [Viraj Sabhaya](https://www.linkedin.com/in/vsabhaya23/)")

def uiSidebarWorkingInfo():
    with st.container():
        st.write("## ℹ️ FAQ")
        with st.expander("How does pdf-GPT: Load and Ask work?", expanded=False):
            # st.write("## How does pdf-GPT: Load and Ask work?")
            st.write(":orange[When you upload a document, it will be divided into smaller chunks and stored in a vector index. A vector index is a special type of database that allows for semantic search and retrieval. Semantic search is a type of search that takes into account the meaning of the words in a query, rather than just the words themselves. This allows pdf-GPT to find the most relevant document chunks for a given question.]")
            st.write(":orange[When you ask a question, pdf-GPT will search through the document chunks and find the most relevant ones using the vector index. Then, it will use GPT3 to generate a final answer. GPT3 is a large language model that can generate text, translate languages, write different kinds of creative content, and answer your questions in an informative way.]")


def clear_submit():
    """
        Clears submit button
    """
    st.session_state["submit"] = False

def main():
    st.set_page_config(page_title="Load & Ask", page_icon="robot_face")
    st.write(
        """<style>
        [data-testid="column"] {
            width: calc(50% - 1rem);
            flex: 1 1 calc(50% - 1rem);
            min-width: calc(50% - 1rem);
        }
        </style>""",
        unsafe_allow_html=True,
    )
    st.title('pdf-GPT: Load & Ask 💬 ')
    st.markdown("---")

    # get API key from USER
    st.write('### 1. Enter your OpenAI API key')
    OPENAI_API_KEY = st.text_input(label="", type='password', placeholder="sk-abcdefghi...", label_visibility="collapsed")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Upload PDF file
    st.write('### 2. Upload your PDF file')
    uploaded_file = st.file_uploader('Upload your PDF file', type=['pdf'], label_visibility="collapsed")

    # Read PDF file and extract text
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            text = read_pdf(uploaded_file)
        # TODO: add support for other file types
        #     text=read_pdf(uploaded_file)
        # elif uploaded_file.name.endswith('.txt'):
        #     text=parse_text(uploaded_file)
        # elif uploaded_file.name.endswith('.docx'):
        #     text=parse_docx(uploaded_file)
        else:
            raise ValueError("File type not supported")
        # st.write(text)
        try:
            with st.spinner("Splitting text into chunks ..."):
                chunks = split_text(text)
        except OpenAIError as e:
            st.error(e._message)
        chunks = split_text(text)
        # st.write(chunks)

        # function to initialize session state
        init_session_state()

        # Ask question
        st.write('### 3. Ask your question'+(f' about {uploaded_file.name}' if uploaded_file else 'uploaded file'))
        user_question = st.text_area(on_change=clear_submit, height=90, label="Ask a question about your PDF here :", placeholder="Type your question here", label_visibility="collapsed")

        button = st.button("Submit!")
        if button or st.session_state.get("submit"):
            if not uploaded_file:
                st.error("Please upload a file!")
            if not user_question:
                st.error("Please ask a question!")
            else:
                st.session_state["submit"] = True
                try:
                    with st.spinner("Searching for answer..."):
                        response = ask_question(user_question, chunks, OPENAI_API_KEY)
                except OpenAIError as e:
                    st.error(e._message)
            #--- with get_openai_callback() as callback:
                # storing past questions and answers
                st.session_state.past.append(user_question)
                st.session_state.generated.append(response)
            #--- print(callback)

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state['generated'][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')

    with st.sidebar:
        uiSidebarInfo()
        uiSidebarWorkingInfo()


if __name__ == '__main__':
    main()