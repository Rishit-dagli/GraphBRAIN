from indigo import *
from indigo.renderer import *
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
page_title = "BMP"
st.set_page_config(page_title=page_title)

import base64



indigo = Indigo()
renderer = IndigoRenderer(indigo)

mols = {   '2083': 'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O',
         '123600': 'CC(C)(C)NC[C@@H](C1=CC(=C(C=C1)O)CO)O',
         '182176': 'CC(C)(C)NC[C@H](C1=CC(=C(C=C1)O)CO)O' }

# array = indigo.createArray()
mol = indigo.loadMolecule(mols['2083'])
# array.arrayAdd(mol)

# for key in mols.keys():
#     print(key, mols[key])
#     mol = indigo.loadMolecule(mols[key])
#     s = "CID=" + key
#     mol.setProperty("grid-comment", s)
#     array.arrayAdd(mol)

# indigo.setOption("render-comment", "Albuterol")
# indigo.setOption("render-comment-position", "top")
# indigo.setOption("render-grid-margins", "40, 10")
# indigo.setOption("render-grid-title-offset", "5")
# indigo.setOption("render-grid-title-property", "grid-comment")
# indigo.setOption("render-background-color", 1.0, 1.0, 1.0)
# indigo.setOption("render-atom-color-property", "color")
# indigo.setOption("render-coloring", True)
# indigo.setOption("render-image-size", 1200, 300)
#
# renderer.renderGridToFile(array, None, 3, "grid.png")
#
# string = renderer.renderToBuffer(mol)
# imgdata = base64.b64decode(string)
# st.image(string)

# renderer.renderToString(mol)


from ccdc.molecule import Molecule
citric = Molecule.from_string("OC(=O)CC(O)(C(=O)O)CC(=O)O")

# components.iframe("https://chemapps.stolaf.edu/jmol/jmol.php?model=CC", scrolling=True, height=1000)
# components.iframe("https://docs.streamlit.io/en/latest", scrolling=True, height=500)


