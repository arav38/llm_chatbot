from langchain_community.llms import Ollama, CTransformers
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
import bs4
from langchain_community.embeddings import OllamaEmbeddings
from fastapi import FastAPI, Request, Form
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from fastapi.templating import Jinja2Templates
from langchain_ollama import OllamaEmbeddings,OllamaLLM
import uvicorn
import requests
import os


os.environ["USER_AGENT"] = "app/1.0"


# Create FastAPI app
# app = FastAPI(title="Langchain Server", description="AI Chat Bot", version="1.0")
app = FastAPI()

# Setup Jinja2 templates
Template = Jinja2Templates(directory="templates")

# Define route for index page
@app.get("/")
async def index(request: Request):
    return Template.TemplateResponse(name="index copy 2.html", context={"request": request})
    
# Set headers
headers = {"User-Agent": "app/1.0"}

# Manually fetch content with requests
url = "https://www.accuweather.com/en/in/pune/204848/weather-forecast/204848"
response = requests.get(url, headers=headers)
response.raise_for_status()  # Ensure the request was successful

# Load document from web and preprocess
loader = WebBaseLoader(
    web_path=("https://www.accuweather.com/en/in/pune/204848/weather-forecast/204848",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="cur-con-weather-card__body"))
    #headers={"User-Agent": "MyApp/1.0"}
)
text_document = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
document = text_splitter.split_documents(text_document)

# Create embeddings and vectorstore
ollama_embedding = OllamaEmbeddings(model="llama2")
db1 = FAISS.from_documents(document, ollama_embedding)

# Define language model
llm = OllamaLLM(model="llama2")

# Create prompt template
prompt = ChatPromptTemplate.from_template("""
Answer question based on user provided input context. Give us detailed steps for the answer.
<context>
{context}
<context>
Question: {input}
""")

# Define answer endpoint
@app.post("/answer")
async def get_retrival_llm_response(request: Request, Question: str = Form(...)):
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = db1.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    result = retrieval_chain.invoke({"input": Question})
    # return {"answer": result["answer"]}
    return Template.TemplateResponse(
            "index copy 2.html",
            {"request": request, "response": result["answer"]},
        )

if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8000)
