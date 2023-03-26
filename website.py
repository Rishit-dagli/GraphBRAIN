# import os
# import random
# import urllib.request
#
# import cv2
# import numpy as np
import streamlit as st
# import tensorflow as tf
# from appwrite.client import Client
# from appwrite.input_file import InputFile
# from appwrite.services.storage import Storage
# from PIL import Image

shape = 224

u = "https://storage.googleapis.com/rishit-dagli.appspot.com/My_project-1_1.png"
page_title = "BMP"


# Set page title and favicon.
st.set_page_config(page_title=page_title, page_icon=u)


# def add_bg_from_url():
#     st.markdown(
#         f"""
#          <style>
#          .stApp {{
#              background-image: url("https://media.discordapp.net/attachments/1043363043947581533/1088974077542281226/30cc0fae-84ce-4013-9dc9-45317c4115b8.jpeg?width=936&height=936");
#              background-attachment: fixed;
#              background-size: cover
#          }}
#          </style>
#          """,
#         unsafe_allow_html=True,
#     )
#
#
# add_bg_from_url()

u = "https://storage.googleapis.com/rishit-dagli.appspot.com/My_project-1_1.png"
# st.image(u, width=150)
col1, mid, col2 = st.columns([7, 1, 25])
with col1:
    st.image(u, width=150)
with col2:
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:35px;">{"Weed Detech"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h2 style="color:#FFFFFF;font-size:20px;">{"The goal of our project is to predict the permeability of molecules through the blood-brain barrier."}</h2>',
        unsafe_allow_html=True,
    )

# Display markdown content


st.markdown(
    f'<h2 style="color:#FFFFFF;font-size:18px;">{"The blood - brain barrier is a protective layer that separates the brain from the rest of the body’s circulatory system. This barrier is highly selective and prevents solutes in the circulating blood from non-selectively crossing into the extracellular fluid of the central nervous system where neurons reside."}</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<h1 style="color:#FFFFFF;font-size:20px;">{"Instructions"}</h1>',
    unsafe_allow_html=True,
)
instruction_1 = "A SMILES string is a representation of the molecule as an ASCII string."
instruction_2 = "When you give us the SMILE string for a molecule, we will predict its Brain Membrane Permeability and output a diagrametical representaion of the molecule."
instruction_3 = "Here are some sample inputs for you to test out our interface-"
st.markdown(
    f'<ul style="color:#FFFFFF"><li>{instruction_1}</li><li>{instruction_2}</li><li>{instruction_3}</li></ul>',
    unsafe_allow_html=True,
)

sample_output_image = "https://storage.googleapis.com/rishit-dagli.appspot.com/My_project-1_1.png"
def load_sample_output():
    # Construct MolView URL
    molview_url = 'https://embed.molview.org/v1/?mode=balls&smiles=' + smiles

    try:
        # Embed 3D rendering of molecule
        # st.components.v1.html(f'<iframe src="{molview_url}" width="600" height="400"></iframe>', height=400)
        st.components.v1.iframe(molview_url, height=400)
    except Exception as e:
        st.error('Error fetching 3D rendering')
        st.error(e)
    # col9.image(sample_output_image, width=500)
    # col10.write("Vfjhbels")


smiles = st.text_input('', placeholder='Input SMILES string here')

gap2, col5, col6, col7, col8, gap3 = st.columns([1, 1, 1, 1, 1, 1])
with col5:
    st.button('Sample 1', on_click=load_sample_output)
with col6:
    st.button('Sample 2', on_click=load_sample_output)
with col7:
    st.button('Sample 3', on_click=load_sample_output)
with col8:
    st.button('Sample 4', on_click=load_sample_output)

col9, col10 = st.columns([1, 100000000000000000000000])

if smiles:
    # Construct MolView URL
    molview_url = f'https://embed.molview.org/v1/?mode=balls&smiles={smiles}'

    try:
        # Embed 3D rendering of molecule
        st.components.v1.html(f'<iframe src="{molview_url}" width="600" height="400"></iframe>', height=400)

    except Exception as e:
        st.error('Error fetching 3D rendering')
        st.error(e)
