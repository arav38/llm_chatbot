from fastapi import FastAPI,Request,Form
from langserve import add_routes
# from langchain.llms import CTransformers
from langchain_community.llms import CTransformers
import uvicorn
from langchain.prompts import PromptTemplate
import streamlit as st
from fastapi.templating import Jinja2Templates


# APP CREATION
app = FastAPI(
    title = "Langchain Server",version= "1.0",description="Chat API"

)

templates = Jinja2Templates(directory="templates")

# Creating the llama model
llama = CTransformers(
    model='Model/llama-2-7b-chat.ggmlv3.q8_0.bin',
    model_type='llama',
    config={'max_new_tokens': 256, 'temperature': 0.01}
)

# Define your prompt template
template = """WRITE BLOG FOR {blog_style} job profile for a topic {input_txt} within {no_words} words."""

prompt = PromptTemplate(
    input_variables=['blog_style', 'input_txt', 'no_words'],
    template=template
)

@app.get('/')
async def index(request:Request):
    return templates.TemplateResponse(name="index.html",context={"request":request})


@app.post('/generate-blog')
async def llmresponce(request:Request,blog_style : str = Form(...), input_txt : str = Form(...),no_words : int =Form(...)):

    formatted_prompt = prompt.format(blog_style=blog_style, input_txt=input_txt, no_words=no_words)

    response = llama(formatted_prompt)

    return templates.TemplateResponse("result.html", {"request": request, "response": response})


if __name__=="__main__":
    uvicorn.run(app=app,host="localhost",port=8000)