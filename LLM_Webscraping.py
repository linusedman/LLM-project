# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 13:51:40 2025

@author: Lovisa
"""

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import io
import os
from urllib.parse import urljoin, urlparse
import re



output_dir = "scraped_pdfs"
os.makedirs(output_dir, exist_ok=True)

base_url = "https://kontrollwiki.livsmedelsverket.se"
search_url = "https://kontrollwiki.livsmedelsverket.se/sok?typ=7"

all_links = []

def change_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "_", name)
    return name.strip()

# loopa över sidnummer – öka intervallet om du vill ha fler sidor
for page in [0, 50, 100, 150, 200, 250]:  # <-- testa först med t.ex. 1–5
    url = f"{search_url}&sidOffset={page}"
    print(f"Hämtar {url} ...")

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    # hitta alla .sok-resultat divar
    results = soup.select("div.sok-resultat")
    if not results:
        print("Inga fler resultat, bryter loopen.")
        break

    for div in results:
        a = div.find("a", href=True)
        if a:
            full_url = urljoin(base_url, a["href"])
            title = a.get_text(strip=True)
            all_links.append((title, full_url))

print("Steg 1 klart!")

#Steg 2: Hämta pdferna för de konsoliderade förordningarna
pdf_links = []
for title, url in all_links:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    section = soup.find("main", id="mainContent")
    if section:
        found_a = section.find("a", href=True)
        for a in found_a:
            konsol_url = found_a["href"]
            konsol_title = found_a.get_text(strip=True)
            #print(f"{konsol_title} -> {konsol_url}")
            pdf_links.append((title, konsol_url))
            
 
print("Klar med steg 2!")         
   
#Steg 3: Spara ner
for title, pdf_url in pdf_links:
    full_url = urljoin("https://eur-lex.europa.eu/", pdf_url)

    parsed = urlparse(full_url)
    filename = change_filename(title) + ".pdf"

    filepath = os.path.join(output_dir, filename)

    try:
        response = requests.get(full_url, stream=True)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except Exception as e:
        print(f"Failed to download {pdf_url}: {e}")