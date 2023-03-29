import streamlit as st
from PIL import Image
from functions_for_streamlit import load_model_in_cache, output_for_button, output_for_string, display_desc_instr, display_goal, set_background_black


page_title = "Graph Brain"
img = Image.open("media/Graph.png")


# Set page title and favicon.
st.set_page_config(page_title=page_title, page_icon=img)

set_background_black()

col1, mid, col2 = st.columns([7, 1, 25])
with col1:
    st.image(img, width=150)
with col2:
    display_goal()

# Display markdown content

tab1, tab2 = st.tabs(["Description and Instructions", "Run the Model"])

with tab1:
    display_desc_instr()


prediction = None
model = load_model_in_cache()


with tab2:
    smiles = st.text_input("", placeholder="Input SMILES string here")

    samples = ["CCC", "CCCC", "CCCCC", "CCCCCC"]
    gap2, col5, col6, col7, col8, gap3 = st.columns([1, 1, 1, 1, 1, 1])

    with col5:
        b1 = st.button("Propane")
    with col6:
        b2 = st.button("Butane")
    with col7:
        b3 = st.button("Pentane")
    with col8:
        b4 = st.button("Hexane")

    output = st.empty()

    if smiles and not (b1 or b2 or b3 or b4):
        prediction = output_for_string(smiles)
    elif b1:
        prediction = output_for_button(0)
    elif b2:
        prediction = output_for_button(1)
    elif b3:
        prediction = output_for_button(2)
    elif b4:
        prediction = output_for_button(3)

    if prediction is not None:
        output.write(
            "Prediction: The Blood-Brain Barrier Permeability of the molecule is"
            f" {prediction}."
        )
