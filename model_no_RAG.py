# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:06:24 2025
Updated on Sun Oct 19 13:32:50 2025

@author: Lovisa
@coauthor: Agnes
@coauthor: Linus
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
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

def response_stream_no_RAG(inputs, history):
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
    if "hej" in user_text and not "hejdå" in user_text:
        yield "Hej, jag är din livsmedelsexpert. Vad kan jag hjälpa dig med?"
        return
    elif "hejdå" in user_text:
        yield "Hejdå! Ha en fortsatt trevlig dag :)"
        return

    history_text = "Fortsätt konversationen.\n\n"
    for user_msg, bot_msg in history:
        history_text += f"Användare: {user_msg}\nAssistent: {bot_msg}\n"

    # Removed RAG

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
    
