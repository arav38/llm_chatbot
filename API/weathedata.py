from langchain_community.document_loaders import WebBaseLoader
import bs4
import pandas as pd


loader = WebBaseLoader(web_path=("https://www.accuweather.com/en/in/pune/204848/weather-forecast/204848",),
                       bs_kwargs=dict(parse_only= bs4.SoupStrainer(class_=("cur-con-weather-card__body"))))

text_document = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter =  RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 200)

document = text_splitter.split_documents(text_document)

# document = document[0].page_content

# line  = [line.strip() for line in document.splitlines() if line.strip()]

# dict = {}
# for l in line:
#     dict['temperature']=line[0]
#     dict['RealFeel']=line[2]
#     dict['RealFeel Shade']=line[6]
#     dict['Wind']=line[8]
#     dict['Wind Gusts']=line[10]
#     dict['Air Quality']=line[12]

# print(dict)


# vector embedding
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

ollama_embedding = OllamaEmbeddings()


def rag_responce(document,query):

    from langchain.vectorstores import FAISS

    db1 = FAISS.from_documents(document,ollama_embedding)

    query = query

    result = db1.similarity_search(query)

    return result[0].page_content


query =  "How's Air Quality ?"

result = rag_responce(document,query=query)

print(result)


#vector embedded and vector store

# from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import FAISS

# ollama_embedding = OllamaEmbeddings()

# db = FAISS.from_documents(document2,ollama_embedding)



# query =  "What is teamprature ?"
# result = db.similarity_search(query)
# print(result[0].page_content)