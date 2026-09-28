"""Streamlit renderers.

`components.py` and `sidebar.py` import streamlit at module scope, which the
phase spec's "app.py is the only file that imports streamlit" line would forbid.
The same spec mandates `render_answer(answer)` and friends with no `st`
parameter, so a renderer cannot receive the module -- the two requirements
cannot both hold. The signatures are the more specific instruction, so the
renderers import streamlit and `app.py` remains the only entry point.
"""
