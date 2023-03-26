import streamlit as st


shape = 224

u = "https://media.discordapp.net/attachments/1043363043947581533/1089608908940775505/GraphBRAIN_logo.png?width=1000&height=1000"
page_title = "Graph Brain"


# Set page title and favicon.
st.set_page_config(page_title=page_title, page_icon=u)


u = "https://media.discordapp.net/attachments/1043363043947581533/1089608908940775505/GraphBRAIN_logo.png?width=1000&height=1000"

col1, mid, col2 = st.columns([7, 1, 25])
with col1:
    st.image(u, width=150)
with col2:
    st.markdown(
        f'<h1 style="color:#FFFFFF;font-size:35px;">{"Graph Brain"}</h1>',
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


def embed_molview(smile):
    molview_url = f'https://embed.molview.org/v1/?mode=balls&smiles={smile}'
    try:
        # Embed 3D rendering of molecule
        st.components.v1.iframe(molview_url, height=400, scrolling=True)
    except Exception as e:
        st.error('Error fetching 3D rendering')
        st.error(e)


smiles = st.text_input('', placeholder='Input SMILES string here')

samples = ['CCC', 'CCCC', 'CCCCC', 'CCCCCC']
gap2, col5, col6, col7, col8, gap3 = st.columns([1, 1, 1, 1, 1, 1])
with col5:
    b1 = st.button('Propane')
with col6:
    b2 = st.button('Butane')
with col7:
    b3 = st.button('Pentane')
with col8:
    b4 = st.button('Hexane')

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
