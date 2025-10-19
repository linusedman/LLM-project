# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:06:24 2025
Updated on Sun Oct 19 13:32:50 2025

@author 1: Lovisa
@author 2: Agnes
@author 3: Linus
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import gradio as gr
import mimetypes
import pdfplumber
from PIL import Image

load_dotenv()
KEY = os.environ.get("GEMINI_API_KEY")
FAISS_INDEX = os.environ.get("FAISS_FOLDER")

client = genai.Client(api_key=KEY)

default_model = "gemini-2.5-flash"

from langchain_community.vectorstores import FAISS  # "db" to store and retrieve embeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

db = FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)  # Load the vector database

def response_stream(inputs, history):
    user_text = ""
    user_images = []

    if isinstance(inputs, dict):
        user_text = inputs.get("text", "").lower()
        files = inputs.get("files", [])

        if files:
            for file_path in files:
                try:
                    mime_type, _ = mimetypes.guess_type(file_path) # Guesses "the filetype based on its filename, path or URL, given by url"

                    if mime_type and mime_type.startswith("image/"):
                        # Handle image input
                        user_images.append(Image.open(file_path))

                    elif mime_type and mime_type.startswith("text/"):
                        # Handle plain text input
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            user_text += "\n" + f.read()

                    elif mime_type == "application/pdf" or file_path.lower().endswith(".pdf"):
                        # Handle PDF input
                        try:
                            with pdfplumber.open(file_path) as pdf:
                                pdf_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                                user_text += "\n" + pdf_text
                        except Exception as e:
                            user_text += f"\n(Kunde inte läsa PDF: {e})"

                    else:
                        # Unsupported file types
                        user_text += "\n(Filtypen stöds inte ännu.)"
                except Exception as e:
                    user_text += "\nFel vid läsning av fil {file_path}: {e}"

    else:
        user_text = inputs.lower()
    
    # special greetings from example file
    if user_text=="hej":
        yield "Hej, jag är din livsmedelsexpert. Vad kan jag hjälpa dig med?"
        
    elif "hejdå" in user_text:
        yield "Hejdå! Ha en fortsatt trevlig dag :)"
        return

    history_text = "Fortsätt konversationen.\n\n"
    for user_msg, bot_msg in history:
        history_text += f"Användare: {user_msg}\nAssistent: {bot_msg}\n"

    # Add context from RAG if any text has been inputted by the user
    if len(user_text) > 0:
        context = db.similarity_search(user_text, k=5)
        history_text += "\n\n" + "Kontext:\n" + "".join([chunk.page_content + "\n Source: " + chunk.metadata["source"] for chunk in context]) + "\n"

    # Add latest user input
    history_text += f"Användare: {user_text}\nAssistent:"
    
    contents = []
    if len(user_images) > 0:
        contents.extend(user_images)

    contents.append(history_text)

    try:
        gemini_stream = client.models.generate_content_stream(
            model=default_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2000,
                system_instruction="Du är en livsmedelsexpert med djup kunskap inom Sveriges och EUs lagar kring livsmedel.\
                      Besvara användarens frågor enligt kontexten, ta hänsyn till alla filer som användaren tillhandahåller.\
                          Var artig och pedagogisk. OM några förordningar finns med i kontexten SÅ avsluta varje meddelande med en lista av de relevanta förordningarna.",
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH",
                        threshold="BLOCK_MEDIUM_AND_ABOVE"
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
        yield "Ursäkta, ett fel uppstod! Kan du upprepa dig snälla!"
        return
    
with gr.Blocks(
    fill_height=True, 
    css="""
        /* Colors commonly used on the website of livsmedelsverket and the kontrollwiki
        orange-crayola: #f2712e;
        white: #ffffff;
        van-dyke: #4c3d38;
        verdigris: #6cacad;
        dark-cyan: #2a898b;
        */

        
        /* Title text */
        .gradio-container .prose h1 {
            color: #f2712e !important;
        }

        /* Whole app background */
        .gradio-container {
            background-color: #6cacad !important;
        }

        /* FUNKAR EJ??? HUR ÄNDRAR MAN CHAT INTERFACE??? */
        #chat-bot .chat-interface { 
            background-color: #f2712e;
            color: #ffffff;
            border-radius: 12px;
        }
        #chat-bot .multimodal-textbox textarea {
            background-color: #ffffff;
            color: #4c3d38;
            border: 2px solid #6cacad;
            border-radius: 8px;
            padding: 6px;
        }
        #info-box .info-container {
            background-color: #f8b88b;
            color: #ffffff;
            padding: 12px;
            border: 2px solid #f2712e;
            border-radius: 10px ;
        }

        #info-box h3 {
            color: #f2712e !important;
        """) as demo:
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Group(elem_id="chat-bot"):
                chatbot = gr.ChatInterface(
                fn=response_stream,
                chatbot=gr.Chatbot(height=700),
                multimodal=True,
                textbox=gr.MultimodalTextbox(
                    placeholder="Fråga mig något, så hjälper jag dig!",  
                    file_count="multiple"),
                title="Din livsmedelsexpert",
                )
        with gr.Column(scale=1):
            with gr.Group(elem_id="info-box"):
                gr.HTML(
                    """
                    <div class="info-container">
                        <h3>Viktig information</h3>
                            <p>Hej kära användare! Jag är din AI-livsmedelsexpert, här för att hjälpa dig med frågor kring livsmedel och relaterade lagar i Sverige och EU.<br><br>  
                        Trots min expertis kan jag ibland göra misstag, så tveka inte att dubbelkolla viktig information. Jag hämtar kontext från konsoliderade 
                        versioner av EU-lagar fram till september 2025, så för de allra senaste uppdateringarna rekommenderar jag att du konsulterar officiella 
                        källor såsom EUR-Lex eller experter inom området.<br><br>  
                        Tack för att du använder vår tjänst!</p>
                    </div>
                    """
                )

if __name__ == "__main__":

    demo.launch(share=False)
