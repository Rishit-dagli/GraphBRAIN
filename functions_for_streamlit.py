"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file defines useful function to run the Streamlit app "Graph Brain".

Copyright and Usage Information
===============================
Copyright 2023 Pranjal Agrawal, Rishit Dagli, Shivesh Prakash and Tanmay Shinde
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import streamlit as st
from model.inference.load_model import load_model
from model.inference.infer import predict
import tensorflow as tf
import python_ta as pyta


def embed_molview(smile: str) -> None:
    """This function embeds a 3D rendering of a molecule with the given SMILES string. It takes a single argument,
    smile, which is a string representing the SMILES notation of the molecule to be rendered.
    If successful, it will render a 3D visualization of the molecule using the MolView service, and embed it using the
    st.components.v1.iframe() function. The iframe has a height of 400 pixels and scrolling enabled.
    If there is an error fetching the 3D rendering, an error message will be displayed using st.error(). The error
    message will include the error object that was raised.
    The function does not return anything, it only displays the 3D rendering or error message.

    Parameters:
        smile: A string representing the SMILES notation of the molecule to be rendered.
    """
    molview_url = f"https://embed.molview.org/v1/?mode=balls&smiles={smile}"
    try:
        # Embed 3D rendering of molecule
        st.components.v1.iframe(molview_url, height=400, scrolling=True)
    except Exception as e:
        st.error("Error fetching 3D rendering")
        st.error(e)


def load_model_in_cache() -> tf.keras.Model:
    """This function loads a TensorFlow Keras model into the Streamlit cache, which is a data store that persists
    across Streamlit app sessions.

    Returns:
        model: a TensorFlow Keras model object that has been loaded into the Streamlit cache.

    Raises:
        This function does not raise any exceptions.

    Note:
        If the model object is already present in the Streamlit cache, this function returns the cached object rather
        than reloading the model."""
    if "model" not in st.session_state:
        st.session_state["model"] = load_model(filename=None)
    return st.session_state["model"]


def output_for_button(
    button_number: int, samples: list, model: tf.keras.Model
) -> float:
    """This function takes in three parameters: button_number, samples, and model. It then uses these
    parameters to return the prediction for the molecule represented by the given button.

    Parameters:
        button_number (int): An integer representing the index of the button clicked.
        samples (list): A list of SMILES strings representing the molecules to predict.
        model (tf.keras.Model): A trained machine learning model to use for prediction.

    Returns:
        prediction (float): A float representing the prediction of the machine learning model for the molecule
        represented by the given button."""
    embed_molview(samples[button_number])
    smiles = samples[button_number]
    smile = []
    smile.append(smiles)
    result = predict(smile, model)
    prediction = result.numpy()[0]
    return prediction


def output_for_string(smiles: str, model: tf.keras.Model) -> float:
    """This function takes a SMILES string representation of a molecule and a trained TensorFlow Keras model, and
    returns a float prediction for the activity of the molecule.

    Parameters:
        smiles (str): A string representing the SMILES of a molecule.
        model (tf.keras.Model): A trained TensorFlow Keras model to use for making predictions.

    Returns:
        prediction (float): The predicted activity score for the molecule represented by the input SMILES string.
    """
    embed_molview(smiles)
    smile = []
    smile.append(smiles)
    result = predict(smile, model)
    prediction = result.numpy()[0]
    return prediction


def display_desc_instr() -> None:
    """This function displays a description and instructions for the website.

    Arguments: None

    Returns: None

    The function displays a header describing the blood-brain barrier, which is followed by a section of instructions
    for the website. The instructions include information on what a SMILES string is, the prediction that the website
    will make based on the input SMILES string, and how to use the sample buttons provided to test the functionality of
    the website. Finally, the function prompts the user to click on the "Run the Model" button to try out the prediction
    feature."""
    description = """The blood - brain barrier is a protective layer that separates the brain from the rest of the 
    body’s circulatory system. This barrier is highly selective and prevents solutes in the circulating blood from 
    non-selectively crossing into the extracellular fluid of the central nervous system where neurons reside."""
    st.markdown(
        f'<h2 style="color:#FFFFFF;font-size:18px;">{description}</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:20px;">{"Instructions"}</h1>',
        unsafe_allow_html=True,
    )
    # with st.expander("Click here to see instructions"):
    inst_1 = "A SMILES string is a representation of the molecule as an ASCII string."
    inst_2 = (
        "When you give us the SMILE string for a molecule, we will predict its"
        " permeability through the blood-brain barrier and render an interactive"
        " molecular structure in 3d."
    )
    inst_3 = (
        "There are 4 sample buttons below the input box to help demonstrate the"
        " functionality of our website"
    )
    inst_4 = "Click on 'Run the Model' to try it out!"
    st.markdown(
        (
            f'<ul style="color:#FFFFFF"><li>{inst_1}</li><li>{inst_2}</li><li>{inst_3}</li><li>{inst_4}</li></ul>'
        ),
        unsafe_allow_html=True,
    )


def display_goal() -> None:
    """Display the title of the project and its goal on the Streamlit website.

    Parameters: None

    Returns: None"""
    goal = """The goal of the project is to predict the permeability of molecules through the blood-brain barrier."""
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:35px;">{"Graph Brain"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h2 style="color:#FFFFFF;font-size:20px;">{goal}</h2>',
        unsafe_allow_html=True,
    )


def set_background_black() -> None:
    """Set the background of the website to black."""
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


# Checking the code for errors using python_ta.
pyta.check_all(
    config={
        "extra-imports": ["streamlit", "PIL", "python_ta"],
        "allowed-io": [],
        "max-line-length": 120,
    },
    output="pyta_output2.txt",
)
