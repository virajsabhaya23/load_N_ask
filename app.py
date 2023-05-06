from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="Load & Ask")
    st.header('Load & Ask 💬 ')

    # Upload PDF file
    pdf_file = st.file_uploader('Upload your PDF file', type=['pdf'])

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
        embeddings = OpenAIEmbeddings()
        retireved_info_knowledge = FAISS.from_texts(chunks, embeddings)

        user_question = st.text_input(label="Ask a question about your PDF here :", placeholder="Type your question here")
        if user_question:
            docs = retireved_info_knowledge.similarity_search(user_question)
            # st.write(docs)

            llms = OpenAI()
            qa_chain = load_qa_chain(llms, chain_type="stuff")
            with get_openai_callback() as callback:
                reponse = qa_chain.run(input_documents = docs, question=user_question)
                print(callback)

            st.write(reponse)

if __name__ == '__main__':
    main()