import requests
from flask import Flask, render_template, redirect, request, url_for
import streamlit as st
import uvicorn
from asgiref.wsgi import WsgiToAsgi


def get_ollma_responce(input_txt):
    responce = requests.post("http://localhost:8000/blogs/invoke",json={
        # "input":{"blog_style":blog_style,"input_txt":input_txt,"no_words":no_words}
        "input":{"input_txt":input_txt}
    })

    return responce.json()['output']['content']

st.title("Langchain Demo with llama model")
input_txt = st.text_input("Blog on ")
no_words = st.number_input("Enter number")
blog_style = st.text_input("Blog style")


# if input_txt | no_words | blog_style:
if input_txt :
    st.write(get_ollma_responce(input_txt))

# app = Flask(__name__)

# @app.route('/', methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         input1 = request.form['input1']  # The topic
#         input2 = request.form["input2"]  # The number of words
#         input3 = request.form['input3']  # The blog style

#         # Pass form data as query parameters using url_for
         
#         return redirect(url_for('success', input1=input1, input2=input2, input3=input3))

#     return render_template(r'\templates\index.html')

# @app.route('/success')
# def success():
#     # Retrieve the query parameters from the URL
#     input1 = request.args.get('input1')
#     input2 = request.args.get('input2')
#     input3 = request.args.get('input3')

#     # Get the Llama response
#     response = get_ollma_responce(input1, input2, input3)
    
#     # Return the generated blog in the response
#     # Return the generated blog along with an OK button to go back to the home page
#     return f"""
#         <h1>Generated Blog:</h1>
#         <p>{response}</p>
#         <form action="/" method="GET">
#             <button type="submit">COPY BUTTON</button>
#         </form>
#     """
# # Wrap the Flask app with ASGI middleware
# asgi_app = WsgiToAsgi(app)

# if __name__=="__main__":
#     uvicorn.run(app=asgi_app,host="localhost",port=8000)
