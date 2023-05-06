import textwrap
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os

def main():
    st.set_page_config(page_title="Load & Ask", page_icon="💬")
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

    st.title('Load & Ask 💬 ')
    st.markdown(
        "This mini-app responses to the questions asks from the PDF uploaded using [OpenAI](https://beta.openai.com/docs/models/overview). You can find the code on [GitHub](https://github.com/virajsabhaya23/load_N_ask) and the author on [LinkedIn](https://www.linkedin.com/in/vsabhaya23/)."
    )

    # Upload PDF file
    pdf_file = st.file_uploader('Upload your PDF file', type=['pdf'])

    # get API key from USER
    OPENAI_API_KEY = st.text_input('Enter your OpenAI API key', type='password')
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if not OPENAI_API_KEY:
        st.warning("Please enter your OpenAI API key.")
        st.stop()

    # Read PDF file and extract text
    if pdf_file is not None:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

        # Split text into sentences/small chunks of text
        splitted_text = CharacterTextSplitter(
            separator='\n',
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitted_text.split_text(text)
        # st.write(chunks)

        # creating embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        retireved_info_knowledge = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input(label="Ask a question about your PDF here :", placeholder="Type your question here")
        if user_question:
            docs = retireved_info_knowledge.similarity_search(user_question)
            # st.write(docs)

            llms = OpenAI(openai_api_key=OPENAI_API_KEY)
            qa_chain = load_qa_chain(llms, chain_type="stuff")
            with get_openai_callback() as callback:
                response = qa_chain.run(input_documents = docs, question=user_question)
                print(callback)

            # st.write(response)
            with st.container():
                with st.expander("ANSWER >", expanded=True):
                    st.info(textwrap.fill(response, width=80))

if __name__ == '__main__':
    main()