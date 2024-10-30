import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
#from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")


st.set_page_config(page_title="Mike's Astro Explorer", page_icon="🚀", layout="wide")

st.markdown("""
<style>
    body {
        color: #e0e0e0;
        background-color: #1d1d1d;
    }
    .stApp {
        background-image: url('https://cdn.pixabay.com/photo/2016/03/18/15/02/ufo-1265186_1280.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    .stButton>button {
        color: #1d1d1d;
        background-color: #e0e0e0;
        border: 2px solid #5c4033;
    }
    .stTextInput>div>div>input {
        color: #1d1d1d;
        background-color: #e0e0e0;
    }
    .stTextArea>div>div>textarea {
        color: #1d1d1d;
        background-color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar :
    st.image('images/astronaut-8061095_1280.webp') #AI_
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()


    options = option_menu(
            "Space Exploration Dashboard", 
            ["Home", "About", "Astro Summarizer", "Astronaut Finder", "Space Discoveries"],
            icons=['house', 'info-circle', 'egg', 'rocket', 'phone', 'newspaper'],
            menu_icon="", 
            default_index=0,
            styles={
                "icon": {"color": "#dec960", "font-size": "20px"},
                "nav-link": {"font-size": "17px", "margin": "5px", "--hover-color": "#262730"},
                "nav-link-selected": {"background-color": "#262730"}
            })

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Tool" :
    st.markdown("""
            <style>
            [data-testid="stAppViewContainer"] > .main {
                background-image: url("https://wallpapercave.com/wp/wp1837539.jpg";
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            </style> """, 
        unsafe_allow_html=True)
    

# Home Page
if options == "Home":
    st.title('Astrophysics & Space Exploration Expert')
    st.write("Welcome! This app provides summaries and insights on astrophysics news, space missions, and discoveries in the cosmos. Whether you're interested in the latest Mars mission, exoplanet findings, or the mysteries of black holes, this tool brings you closer to the universe.")
    st.write("## What the Tool Does")
    st.write("Our tool extracts critical information from reports and articles related to astrophysics and space exploration, presenting them in structured summaries.")
    st.write("## Key Features")
    st.write("- **Efficient Summarization:** Focuses on core details like mission objectives, celestial events, and scientific findings.")
    st.write("- **Astro-Contextual Analysis:** Provides additional context by linking current discoveries with historical data and scientific implications.")
    st.write("- **Ideal for Researchers and Enthusiasts:** For those who want detailed, unbiased summaries without reading through lengthy articles.")

# About Us Page
elif options == "About":
    st.title('About Us')
    st.write("## Meet the Team")
    st.write("Our team is passionate about making space exploration and astrophysics accessible to everyone. With backgrounds in AI, astrophysics, and science communication, we aim to bridge the gap between complex scientific information and the public.")

# Astro Summarizer Tool
elif options == "Astro Summarizer":
    st.title('Astro News Summarizer')
    article_text = st.text_area("Enter Astrophysics/Space Exploration Article", placeholder="Paste the article text here...")
    generate_summary = st.button("Generate Summary")

    if generate_summary:
        with st.spinner("Generating Summary..."):
            System_Prompt = """
            You are an expert in astrophysics and space exploration. Summarize the article by focusing on:
            1. Core event or discovery (e.g., a new exoplanet discovery, a Mars rover mission update).
            2. Scientific context and importance.
            3. Key figures involved, like scientists, astronauts, or space agencies.
            4. Data or quotes that provide credibility.
            5. Future implications or potential advancements.

            Follow this format:
            - **Headline**
            - **Lead**: Summary of the main discovery/event.
            - **Significance**: Why it matters in the context of space exploration.
            - **Details**: Key points, relevant quotes, or data.
            - **Historical Context**: Any related past events or missions.
            - **Future Outlook**: Possible developments or research directions.
            """
            user_message = article_text
            struct = [{'role' : 'system', 'content' : System_Prompt}]
            struct.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini", messages = struct)
            response = chat.choices[0].message.content
            struct.append({"role": "assistant", "content": response})
            st.success("Insight generated successfully!")
            st.subheader("Summary : ")
            st.write(response)

# Astronaut Finder Page
elif options == "Astronaut Finder":
    st.title('Astronaut Insights')
    astronaut_name = st.text_input("Enter the Astronaut's Name:")
    search_button = st.button("Search")

    if search_button:
        with st.spinner("Searching for details..."):
            prompt = f"Provide an in-depth profile for astronaut {astronaut_name}, covering their background, notable missions, and contributions to space exploration."
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are an expert in space exploration."}, {"role": "user", "content": prompt}]
            )
            astronaut_profile = response.choices[0].message.content
            st.subheader("Astronaut Profile")
            st.write(astronaut_profile)

#Space Discoveries Page
elif options == "Space Discoveries":
    st.title("Explore Recent Space Discoveries")
    st.write("Stay updated on the latest findings in the cosmos. From newly discovered galaxies to the latest in dark matter research, our Deep Space Discoveries section provides condensed insights for enthusiasts and researchers alike.")

# Footer and Contact Information
with st.container():
    st.write("### Contact Us")
    mention(label="LinkedIn", url="https://www.linkedin.com")
    mention(label="GitHub", url="https://github.com")
