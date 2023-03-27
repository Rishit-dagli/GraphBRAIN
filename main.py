import requests
import streamlit as st
from model.inference.load_model import load_model
from model.inference.infer import inference
from streamlit_lottie import st_lottie

logo = "http://store.rishit.tech/GraphBRAIN_logo.png"
page_title = "Graph Brain"


# Set page title and favicon.
st.set_page_config(page_title=page_title, page_icon=logo)

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

col1, mid, col2 = st.columns([7, 1, 25])
with col1:
    st.image(logo, width=150)
with col2:
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:35px;">{"Graph Brain"}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<h2 style="color:#FFFFFF;font-size:20px;">{"""The goal of our project is to predict whether a molecule is permeable through the blood-brain barrier."""}</h2>',
        unsafe_allow_html=True,
    )

# Display markdown content

tab1, tab2 = st.tabs(["Description and Instructions", "Model"])

with tab1:
    st.markdown(
        f'<h2 style="color:#FFFFFF;font-size:18px;">{"The blood - brain barrier is a protective layer that separates the brain from the rest of the body’s circulatory system. This barrier is highly selective and prevents solutes in the circulating blood from non-selectively crossing into the extracellular fluid of the central nervous system where neurons reside."}</h2>',
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
    instruction_2 = "When you give us the SMILE string for a molecule, we will predict whether it is permeable through the blood-brain barrier and render an interactive molecular structure in 3d."
    instruction_3 = "There are 4 sample buttons below the input box to help demonstrate the functionality of our website"
    instruction_4 = (
        "NOTE: YOU MAY HAVE TO SCROLL DOWN TO SEE THE 3D RENDERING OF THE MOLECULE"
    )
    st.markdown(
        f'<ul style="color:#FFFFFF"><li>{instruction_1}</li><li>{instruction_2}</li><li>{instruction_3}</li><li>{instruction_4}</li></ul>',
        unsafe_allow_html=True,
    )


def embed_molview(smile):
    molview_url = f"https://embed.molview.org/v1/?mode=balls&smiles={smile}"
    try:
        # Embed 3D rendering of molecule
        st.components.v1.iframe(molview_url, height=400, scrolling=True)
    except Exception as e:
        st.error("Error fetching 3D rendering")
        st.error(e)


prediction = None
model = load_model(filename=None)

url = requests.get("https://assets1.lottiefiles.com/packages/lf20_q8ND1A8ibK.json")
url_json = dict()
if url.status_code == 200:
    url_json = url.json()
else:
    print("Error in the URL")

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
        embed_molview(smiles)
    elif b1:
        embed_molview(samples[0])
    elif b2:
        embed_molview(samples[1])
    elif b3:
        embed_molview(samples[2])
    elif b4:
        embed_molview(samples[3])

    gif_runner = st_lottie(url_json)
    result = inference(smiles, model)
    gif_runner.empty()
    prediction = int(result)
    if prediction is not None:
        output.write(
            f'Prediction: The molecule is {"permeable" if prediction == 1 else "not permeable"} through the blood-brain barrier.'
        )
