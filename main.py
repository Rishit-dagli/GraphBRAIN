"""CSC111 Winter 2023 Final Project: Graph Brain

This Python file contains code for the Streamlit app "Graph Brain". The app allows the user to input a SMILES
string representing a molecule, and predicts the molecule's Blood-Brain Barrier Permeability. The app also includes
buttons for predetermined SMILES strings.

The app has the following dependencies:
streamlit
PIL
python_ta

The app imports the following functions from a separate file called functions_for_streamlit.py:
load_model_in_cache: loads the machine learning model into the Streamlit cache.
output_for_button: processes a user input from a button and returns the prediction.
output_for_string: processes a user input from a text input and returns the prediction.
display_desc_instr: displays the description and instructions for the app.
display_goal: displays the goal of the app.
set_background_black: sets the background color of the app to black.

Page Configuration
The app sets the page title and favicon using the set_page_config() method from the Streamlit library. It also sets the
background color to black using the set_background_black() function.

Layout
The app uses the columns() method from the Streamlit library to create a layout with two columns. The left column
contains the Graph Brain logo and the app goal, and the right column contains two tabs: "Description and Instructions"
and "Run the Model". The tabs() method is used to create the tabs.

User Input
The user can input a SMILES string using a text input field, and can also click on one of four buttons representing
predetermined SMILES strings.

Prediction
When the user inputs a SMILES string or clicks on a button, the app uses the machine learning model to predict the
Blood-Brain Barrier Permeability of the molecule. The prediction is displayed in the output section of the app.

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

import os
import math
import streamlit as st
from PIL import Image
import python_ta as pyta
from functions_for_streamlit import (
    load_model_in_cache,
    output_for_button,
    output_for_string,
    display_desc_instr,
    display_goal,
    set_background_black,
    embed_molview,
    information,
)


path = os.path.dirname(os.path.abspath(__file__))

# Set page title and logo.
page_title = "Graph Brain"
img = Image.open(os.path.join(path, "media", "Graph.png"))


# Configure the title and favicon of the app.
st.set_page_config(page_title=page_title, page_icon=img)


# Set background color to black.
set_background_black()


# Using the columns feature of Streamlit to display the logo and goal.
col1, mid, col2 = st.columns([7, 1, 25])
with col1:
    st.image(img, width=150)
with col2:
    display_goal()


# Creating separate tabs for the description and instructions and the model.
tab1, tab2 = st.tabs(["Description and Instructions", "Run the Model"])


# Displaying the description and instructions in the first tab.
with tab1:
    display_desc_instr()


# Setting the variable prediction to None and loading the model into the cache to optimize the user experience.
prediction = None
model = load_model_in_cache()


# Displaying the model in the second tab.
with tab2:
    # Creating a text input field for the user to input a SMILES string.
    smiles = st.text_input("", placeholder="Input SMILES string here")

    # Creating samples for the users to try out.
    samples = [
        "Fc1ccccc1C2=NCC(=S)N(CC(F)(F)F)c3ccc(Cl)cc23",
        "CC(C)N=C1C=C2N(c3ccc(Cl)cc3)c4ccccc4N=C2C=C1Nc5ccc(Cl)cc5",
        "NC(=O)OCCCc1ccccc1",
        "NCCc1ccc(O)c(O)c1",
    ]

    # Creating columns for the buttons.
    gap2, col5, col6, col7, col8, gap3 = st.columns([0.3, 1, 1, 1, 1, 0.3])

    # Creating buttons for the samples.
    with col5:
        b1 = st.button("Quazepam")
    with col6:
        b2 = st.button("Clofazimine")
    with col7:
        b3 = st.button("Phenprobamate")
    with col8:
        b4 = st.button("Dopamine")

    # Creating an output section for the prediction.
    output1, output2, output3 = st.empty(), st.empty(), st.empty()

    # Checking if the user has input a SMILES string or clicked on a button.
    if smiles and not (b1 or b2 or b3 or b4):
        try:
            molview, description, prediction = (
                smiles,
                -1,
                output_for_string(smiles, model),
            )
        except:
            molview, description, prediction = None, -1, -100
    elif b1:
        molview, description, prediction = None, 0, output_for_button(0, samples, model)
    elif b2:
        molview, description, prediction = None, 1, output_for_button(1, samples, model)
    elif b3:
        molview, description, prediction = None, 2, output_for_button(2, samples, model)
    elif b4:
        molview, description, prediction = None, 3, output_for_button(3, samples, model)

    # Displaying the prediction in the output section.
    if prediction is not None:
        if prediction == -100:
            output1.write("Error: Invalid SMILES string.")
            output2.write("")
        else:
            output1.write(
                "Prediction: The Blood-Brain Barrier Permeability of the molecule is"
                f" {math.floor((float(prediction))*100)/100}."
            )
            output2.write(
                f"Since this value is {'less than 0.3' if float(prediction) < 0.3 else 'greater than 0.3'}, "
                f"the molecule is {'not' if float(prediction) < 0.3 else ''} permeable to the Blood-Brain Barrier."
            )
            if molview is not None:
                embed_molview(molview)

            if 0 <= description <= 3:
                output3.write(information[description])


# Checking the code for errors using python_ta.
pyta.check_all(
    config={
        "extra-imports": ["streamlit", "PIL", "python_ta", "functions_for_streamlit", "math", "os"],
        "allowed-io": [],
        "max-line-length": 120,
        "disable": ["forbidden-top-level-code",
                    "invalid-name",
                    "forbidden-global-variables"],
    },
    output=os.path.join(path, "pyta_outputs", "main.html"),
)
