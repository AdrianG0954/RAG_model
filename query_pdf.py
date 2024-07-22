import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
import openai
from collections import defaultdict

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']
CHROMA_PATH = "chromaPDF"

TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer this question based on the above context: {question}
""" 

def processDB():
    query_text = ""
    while True:
        query_text = input("\n\nEnter your query: ")

        if query_text.strip() == "exit":
            break
    
        embedding_func = OpenAIEmbeddings()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_func)

        results = db.similarity_search_with_score(query_text, k=5)

        
        text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
        prompt_template = ChatPromptTemplate.from_template(TEMPLATE)
        prompt = prompt_template.format(context=text, question=query_text)

        mp = defaultdict(set)
        chat = ChatOpenAI()
        response = chat.invoke(prompt)
        for doc, _score in results:
            mp[doc.metadata.get("source", None)].add(doc.metadata.get("page", None) + 1)
        print(mp)

        sources = []

        for key, value in mp.items():
            sources.append(f'Source: {key}, Pages: {value} | ')
        formatted_response = f'\n\nResponse: {response.content}\n\n{" ".join(sources)}'

        print(formatted_response)
    

if __name__ == "__main__":
    processDB()