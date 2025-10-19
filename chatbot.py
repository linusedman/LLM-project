# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:06:24 2025

@author: Lovisa
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import gradio as gr
load_dotenv()
KEY = os.environ.get("GEMINI_API_KEY")

client = genai.Client(api_key=KEY)  # here you can also pass the api_key directly using os.environ['GEMINI_API_KEY']

default_model = "gemini-2.5-flash"

from langchain_community.vectorstores import FAISS  # "db" to store and retrieve embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

db = FAISS.load_local("faiss_index_all_sv_cs1000", embeddings, allow_dangerous_deserialization=True)  # Load the vector database

def response_stream(inputs, history):
    user_text = ""
    user_image = None
        
    if type(inputs) == dict:
        user_text = inputs.get("text", "").lower()
        files = inputs.get("files", [])
        if files:  # only uses the first uploaded image
            user_image = Image.open(files[0])
    else:
        user_text = inputs.lower()
    
    # special greetings from example file
    if "hej" in user_text and not "hejdå":
        yield "Hej, jag är din livsmedelsexpert. Vad kan jag hjälpa dig med?"
        return
    elif "hejdå" in user_text:
        yield "Hejdå! Ha en fortsatt trevlig dag :)"
        return

    history_text = "Fortsätt konversationen.\n\n"
    for user_msg, bot_msg in history:
        history_text += f"Användare: {user_msg}\nAssistent: {bot_msg}\n"

    # Add context from RAG
    context = db.similarity_search(user_text, k=5)
    history_text += "\n\n" + "Kontext:\n" + "".join([chunk.page_content + "\n Source: " + chunk.metadata["source"] for chunk in context]) + "\n"
    print(history_text)

    history_text += f"Användare: {user_text}\nAssistent:"
    
    contents = []
    if user_image is not None:
        contents.append(user_image)

    contents.append(history_text)

    try:
        gemini_stream = client.models.generate_content_stream(
            model=default_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2000,
                system_instruction="Du är en livsmedelsexpert med djup kunskap inom Sveriges och EUs lagar kring livsmedel. Besvara användarens frågor enligt kontexten. Var artig och pedagogisk, och avsluta varje meddelande med en lista av de relevanta förordningarna.",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE"
                    )
                ]
            )
        )

        # Yield chunks for live updates
        partial_response = ""
        for chunk in gemini_stream:
            if chunk.text:
                partial_response += chunk.text
                yield partial_response
                

    except Exception as e:
        # Handle streaming failure without crashing the chatbot
        yield "Ursäkta, kan du upprepa dig snälla :)"
        return
    
with gr.Blocks(fill_height=False, theme=gr.themes.Citrus(primary_hue=gr.themes.colors.amber, secondary_hue=gr.themes.colors.amber), css="""
        /* Whole app background */
        .gradio-container {
        background-color: #cd4c06 !important;
        }

        /* Title text */
        .gradio-container .prose h1 {
        color: white !important;
        }

        /* Chat area background */
        .gr-chatbot {
        background-color: #ffffff !important; /* white */
        }
        """) as demo:
    chatbot = gr.ChatInterface(
        fn=response_stream,
        multimodal=True,
        title="Din livsmedelsexpert",
    )

# From lab
# This part closes the demo server if it is already running (which
# happens easily in notebooks) and prevents you from opening multiple
# servers at the same time.
#if "demo" in locals() and demo.is_running:
#    demo.close()
#----------------------------------------------------------------------------------
if __name__ == "__main__":

    demo.launch()
