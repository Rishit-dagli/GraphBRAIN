import streamlit as st
from model.inference.load_model import load_model
from model.inference.infer import predict
from PIL import Image
import tensorflow as tf


def embed_molview(smile: str) -> None:
    """Embed a 3D rendering of the molecule with the given SMILES string."""
    molview_url = f"https://embed.molview.org/v1/?mode=balls&smiles={smile}"
    try:
        # Embed 3D rendering of molecule
        st.components.v1.iframe(molview_url, height=400, scrolling=True)
    except Exception as e:
        st.error("Error fetching 3D rendering")
        st.error(e)


def load_model_in_cache() -> tf.keras.Model:
    """Load the model into the Streamlit cache."""
    if "model" not in st.session_state:
        st.session_state["model"] = load_model(filename=None)
    return st.session_state["model"]


def output_for_button(button_number: int) -> float:
    """Return the prediction for the molecule represented by the given button."""
    embed_molview(samples[button_number])
    smiles = samples[button_number]
    smile = []
    smile.append(smiles)
    result = predict(smile, model)
    prediction = result.numpy()[0]
    return prediction


def output_for_string(smiles: str) -> float:
    """Return the prediction for the molecule represented by the given string."""
    embed_molview(smiles)
    smile = []
    smile.append(smiles)
    result = predict(smile, model)
    prediction = result.numpy()[0]
    return prediction


def display_desc_instr():
    st.markdown(
        (
            f'<h2 style="color:#FFFFFF;font-size:18px;">{"The blood - brain barrier is a protective layer that separates the brain from the rest of the body’s circulatory system. This barrier is highly selective and prevents solutes in the circulating blood from non-selectively crossing into the extracellular fluid of the central nervous system where neurons reside."}</h2>'
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:20px;">{"Instructions"}</h1>',
        unsafe_allow_html=True,
    )
    # with st.expander("Click here to see instructions"):
    instruction_1 = (
        "A SMILES string is a representation of the molecule as an ASCII string."
    )
    instruction_2 = (
        "When you give us the SMILE string for a molecule, we will predict its"
        " permeability through the blood-brain barrier and render an interactive"
        " molecular structure in 3d."
    )
    instruction_3 = (
        "There are 4 sample buttons below the input box to help demonstrate the"
        " functionality of our website"
    )
    instruction_4 = "Click on 'Run the Model' to try it out!"
    st.markdown(
        (
            f'<ul style="color:#FFFFFF"><li>{instruction_1}</li><li>{instruction_2}</li><li>{instruction_3}</li><li>{instruction_4}</li></ul>'
        ),
        unsafe_allow_html=True,
    )


def display_goal():
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:35px;">{"Graph Brain"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            f'<h2 style="color:#FFFFFF;font-size:20px;">{"""The goal of the project is to predict the permeability of molecules through the blood-brain barrier."""}</h2>'
        ),
        unsafe_allow_html=True,
    )


def set_background_black():
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-color:black
         }}
         </style>
         """,
        unsafe_allow_html=True,
    )