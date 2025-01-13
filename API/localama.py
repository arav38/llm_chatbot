from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from langchain_ollama import OllamaLLM
import uvicorn

# APP CREATION
app = FastAPI(
    title="Langchain Server", version="1.0", description="Chat API"
)

templates = Jinja2Templates(directory="templates")

# Creating a prompt template
template = ChatPromptTemplate.from_messages([
    ('system', 'Please answer all question ask by user.'),
    ('user', 'Question: {question}')
])


@app.get('/')
async def index(request: Request):
    return templates.TemplateResponse(name='index copy.html', context={'request': request})


@app.post('/generate-blog')
async def llm_response(request: Request, input_txt: str = Form(...)):
    # Format the prompt using the template
    formatted_prompt = template.format(question=input_txt)
    
    # Creating the llama model
    llama = OllamaLLM(model='llama3.1')

    try:
        # Use llama.invoke() or llama.call() depending on API method
        response = llama.invoke(formatted_prompt)

        print(response)
    except Exception as e:
        # Handle any errors that occur
        response = f"An error occurred: {e}"

    # Return the response to the HTML template
    return templates.TemplateResponse("result.html", {"request": request, "response": response})


if __name__ == "__main__":
    uvicorn.run(app=app, host="localhost", port=8001)
