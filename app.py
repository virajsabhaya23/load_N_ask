import textwrap
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

        if not OPENAI_API_KEY:
            st.warning("Please enter your OpenAI API key.")
            st.stop()

        # creating embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        retireved_info_knowledge = FAISS.from_texts(chunks, embeddings)

        #list to store retrieved info
        qna_list = []

        # streamlit-chat to store past questions
        if 'generated' not in st.session_state:
            st.session_state['generated'] = []
        if 'past' not in st.session_state:
            st.session_state['past'] = []

        # Ask question
        user_question = st.text_input(label="Ask a question about your PDF here :", placeholder="Type your question here")
        if user_question:
            docs = retireved_info_knowledge.similarity_search(user_question)
            # st.write(docs)
            llms = OpenAI(openai_api_key=OPENAI_API_KEY)
            qa_chain = load_qa_chain(llms, chain_type="stuff")
            with get_openai_callback() as callback:
                response = qa_chain.run(input_documents = docs, question=user_question)
                # storing past questions and answers
                st.session_state.past.append(user_question)
                st.session_state.generated.append(response)
                print(callback)

            # add Q & A to list
            # qna_list.append((user_question, response))

            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state['generated'][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i)+'_user')


if __name__ == '__main__':
    main()