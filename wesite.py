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

placeholder_text = "Input SMILES string here"
changed = False

# Set page title and favicon.
st.set_page_config(page_title=page_title, page_icon=u)


def add_bg_from_url():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("https://media.discordapp.net/attachments/1043363043947581533/1088974077542281226/30cc0fae-84ce-4013-9dc9-45317c4115b8.jpeg?width=936&height=936");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


add_bg_from_url()

u = "https://storage.googleapis.com/rishit-dagli.appspot.com/My_project-1_1.png"
# st.image(u, width=150)
col1, mid, col2 = st.columns([7, 1, 25])
with col1:
    st.image(u, width=150)
with col2:
    st.markdown(
        f'<h1 style="color:#000000;font-size:35px;">{"Weed Detech"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h2 style="color:#000000;font-size:20px;">{"The goal of our project is to predict the permeability of molecules through the blood-brain barrier."}</h2>',
        unsafe_allow_html=True,
    )

# Display markdown content


st.markdown(
    f'<h2 style="color:#000000;font-size:18px;">{"The blood - brain barrier is a protective layer that separates the brain from the rest of the body’s circulatory system. This barrier is highly selective and prevents solutes in the circulating blood from non-selectively crossing into the extracellular fluid of the central nervous system where neurons reside."}</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<h1 style="color:#000000;font-size:20px;">{"Instructions"}</h1>',
    unsafe_allow_html=True,
)
instruction_1 = "A SMILES string is a representation of the molecule as an ASCII string."
instruction_2 = "When you give us the SMILE string for a molecule, we will predict its Brain Membrane Permeability and output a diagrametical representaion of the molecule."
instruction_3 = "Here are some sample inputs for you to test out our interface-"
st.markdown(
    f'<ul style="color:#000000"><li>{instruction_1}</li><li>{instruction_2}</li><li>{instruction_3}</li></ul>',
    unsafe_allow_html=True,
)

sample_output_image = "https://storage.googleapis.com/rishit-dagli.appspot.com/My_project-1_1.png"
def load_sample_output():
    # st.markdown(f"""
    # <img src={sample_output_image} style="top: 1000px">
    # """,
    # unsafe_allow_html=True,)
    col9.image(sample_output_image, width=500)
    col10.write("Vfjhbels")


title = st.text_input('', placeholder='Input SMILES string here', on_change=load_sample_output)

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



# file = st.file_uploader("", type=["jpg", "png"])


# def load_to_appwrite():
#     if file is None:
#         st.write("Please upload an image.")
#     else:
#         client = Client()
#         (
#             client.set_endpoint("http://34.139.148.58/v1")
#             .set_project("637009ba1cc4e478f2ac")
#             .set_key(
#                 "f3c9e9b0ed0b3fb2b6029687884b332f7df170546104aff0f9ef0cb10829de1235e2bde03a7b2f1f97e0efa78426fa444da66fa8e750caa9a81a5dc107b68ed785bb7ef3d2149e5f162621a75d1a83749657ddfad7336f2b4aaed96d25ad2b59694d898b0ed9773f7ea1a7237cccc5d4916f4ae96c2fd730b16cc8fd69758a4b"
#             )
#         )
#         storage = Storage(client)
#         result = storage.create_file(
#             "637009d1ea462ff0d224", "unique()", InputFile.from_path("img.jpg")
#         )
#         st.markdown(
#             f'<h1 style="color:#000000;font-size:15px;">{"Thank you for participating in improving and personalizing Weed Detech. The data you put in is anonymized before being used for training."}</h1>',
#             unsafe_allow_html=True,
#         )


# def load_model():
#     if "model" not in st.session_state:
#         urllib.request.urlretrieve(
#             "https://github.com/Shivesh777/weed-detech/releases/download/model-weights/model.h5",
#             "model.h5",
#         )
#         st.session_state["model"] = tf.keras.models.load_model("model.h5")
#     return st.session_state["model"]


# if file is None:
#     pass
# else:
#     img = Image.open(file)
#     img = img.save("img.jpg")

#     image = cv2.imread("img.jpg")
#     image = cv2.resize(image, (shape, shape))
#     image_1 = np.reshape(image, (1, shape, shape, 3))
#     pred = load_model().predict(image_1)
#     startX = int(pred[1][0][0] * 224)
#     startY = int(pred[1][0][1] * 224)
#     endX = int(pred[1][0][2] * 224)
#     endY = int(pred[1][0][3] * 224)
#     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
#     Image.fromarray(image).save("img.jpg")
#     st.image("img.jpg", use_column_width=True)
#     st.button(label="Opt into testing", on_click=load_to_appwrite)
