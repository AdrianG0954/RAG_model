import argparse
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import openai

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chroma"

TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer this question based on the above context: {question}
""" 

def processDB():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("query_text", type=str, help="The text you would like to search for.")
    # args = parser.parse_args()
    # query_text = args.query_text
    query_text = ""
    while True:
        query_text = input("\n\nEnter your query: ")

        if query_text.strip() == "exit":
            break
    
        embedding_func = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

        results = db.similarity_search_with_relevance_scores(query_text, k=5)
        if len(results) == 0 or results[0][1] < 0.7:
            print("No relevant results found")
            continue
        
        text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
        prompt = prompt_template.format(context=text, question=query_text)

        chat = ChatOpenAI()
        response = chat.invoke(prompt)
        source = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f'\n\nResponse: {response.content}\n\nSources:{set(source)}'

        print(formatted_response)
    

if __name__ == "__main__":
    processDB()