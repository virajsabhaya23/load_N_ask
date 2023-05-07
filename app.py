import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import os

def read_pdf(userFile):
    """
        Reads a PDF file and extracts its text
        :param userFile: PDF file uploaded by user
        :return: text extracted from PDF file
    """
    pdf_reader = PdfReader(userFile)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def split_text(text):
    """
        Splits a text into smaller chunks
        :param text: text to be split
        :return: list of chunks
    """
    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return chunks

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
    # about this app and instructions CONTAINER
    with st.container():
        with st.expander("ℹ️ About this app & Instructions"):
            # divide into two columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    "This app uses [OpenAI](https://beta.openai.com/docs/models/overview)'s API to answer questions about your PDF file. You can find the code on [GitHub](https://github.com/virajsabhaya23/load_N_ask) and the author on [LinkedIn](https://www.linkedin.com/in/vsabhaya23/)."
                )
            with col2:
                st.markdown("1. Upload your PDF file.\n 2. Enter OpenAI API key. \n 3. Ask your question.")
    st.markdown("---")

    # Upload PDF file
    pdf_file = st.file_uploader('Upload your PDF file', type=['pdf'])

    # get API key from USER
    OPENAI_API_KEY = st.text_input('Enter your OpenAI API key', type='password')
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Read PDF file and extract text
    if pdf_file is not None:
        text=read_pdf(pdf_file)
        # st.write(text)
        chunks = split_text(text)
        # st.write(chunks)

        if not OPENAI_API_KEY:
            st.warning("Please enter your OpenAI API key.")
            st.stop()

        # creating embeddings
        knowledge = create_embeddings(OPENAI_API_KEY, chunks)

        # function to initialize session state
        init_session_state()

        # Ask question
        user_question = st.text_input(label="Ask a question about your PDF here :", placeholder="Type your question here")
        if user_question:
            with st.spinner("Searching for answer..."):
                docs = knowledge.similarity_search(user_question)
                # st.write(docs)
                llms = OpenAI(openai_api_key=OPENAI_API_KEY)
                qa_chain = load_qa_chain(llms, chain_type="stuff")
                with get_openai_callback() as callback:
                    response = qa_chain.run(input_documents = docs, question=user_question)
                    # storing past questions and answers
                    st.session_state.past.append(user_question)
                    st.session_state.generated.append(response)
                    print(callback)
            st.success("Done!")

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state['generated'][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')


if __name__ == '__main__':
    main()