from dotenv import load_dotenv
import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import CSVLoader
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain



load_dotenv()


llm = ChatOpenAI(model_name="gpt-4", api_key=api_key=st.secrets["api_key"], temperature=0.4 )



instructor_embedding=HuggingFaceBgeEmbeddings()
vectordb_file_path='faiss_index'
def create_vector_db():
    loader=CSVLoader(file_path='data.csv', source_column='prompt')
    data=loader.load()
    vectordb=FAISS.from_documents(documents=data, embedding=instructor_embedding)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():    
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embedding, allow_dangerous_deserialization=True)
    retrieval= vectordb.as_retriever(score_threshold=0.7)

    prompt_template=""" given the following context and a question, generate an answer based on contentprompt_selector.
    In answer try to give as much detail as possible and also give a motivational quote in the end of answer.
    if answer is not found kindly response them using other resources but remember reply only those query related to career or choosing a course or learning a new skill. If someone ask about something else then reply "Please ask a question related to career or choosing a course".
    context: {context}
    question: {question}
    """
    prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain=RetrievalQA.from_chain_type(
        chain_type="stuff",
        llm=llm,
        retriever=retrieval,
        return_source_documents=True,
        chain_type_kwargs={"prompt" : prompt}) 
    return chain

if __name__ == "__main__":
    chain=get_qa_chain()

    print(chain.invoke("I am a student and I want to learn a new skill. Can you suggest me a course?"))

   
