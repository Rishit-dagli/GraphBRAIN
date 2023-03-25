import streamlit as st

st.title('Brain Membrane Permeability Prediction')

# Input SMILES string
smiles = st.text_input('Enter SMILES string of compound')

# Check if input is valid
if smiles:
    # Construct MolView URL
    molview_url = f'https://embed.molview.org/v1/?mode=balls&smiles={smiles}'

    try:
        # Embed 3D rendering of molecule
        st.components.v1.html(f'<iframe src="{molview_url}" width="600" height="400"></iframe>', height=400)

    except Exception as e:
        st.error('Error fetching 3D rendering')
        st.error(e)

else:
    st.warning('Please enter a SMILES string')
