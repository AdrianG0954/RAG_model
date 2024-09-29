from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import openai
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")
DATA_PATH = 'data/pdfs'

TEMPLATE = """
Answer the question based only on the following context:

{context}

---

fufill this request: {request}
""" 

def load_pdfs():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load_and_split()
    print("\nStarted...\n")
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    print(len(documents))

    summary = chain.invoke({"input_documents": documents})

    print("\nDONE!\n")
    return summary

def query(summary):

    prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
    prompt = prompt_template.format(context=summary, request="summarize the content")

    response = llm.invoke(prompt)

    print(response.content)
    

    
if __name__ == "__main__":
    summary = load_pdfs()
    query(summary)