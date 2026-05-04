'''
Module: hitl_ui.py

User interface for Human in the Loop Verification
'''
import streamlit as st
from embed import load_jsonl

results_path = 'results/resnet50_results.jsonl'


def extract_info(path: str) -> str:
    '''Extracts the file name and mods (if applicable) from image path

    Args:
        path(str): path to an index or query image
    Returns:
        str: image name and mods formatted for future markdown rendering
    '''
    img_name = (path.rsplit('/', 1))[-1]  # extract image name from full image path
    img_name = (img_name.rsplit('.', 1))[0]  # remove the extension

    img_name_list = img_name.split('_')

    if len(img_name_list) == 2:  # image is an index
        name = '_'.join(img_name_list)
        return f'**Name**: {name}'
    else:
        name = '_'.join(img_name_list[0:2])
        mods = '_'.join(img_name_list[2:])
        return f'**Name**: {name}  **Mods**: {mods}'


# st.cache_data cannot be applied to functions defined in other modules via @decorator syntax.
# Calling it directly here wraps load_jsonl with Streamlit caching without redefining it.
load_results = st.cache_data(load_jsonl)


# --- session state ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# --- load all records (runs once, then cached) ---
results = load_results(results_path)
total = len(results)

# --- fetch current record ---
result = results[st.session_state.idx]
query_path = result['query_path']

# --- layout ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader('Query')
    st.image(query_path)
    st.markdown(extract_info(query_path))

with col2:
    st.subheader('Top Matches')
    for match in result['matches']:
        index_path = match['index_path']
        st.image(index_path)
        st.markdown(f"**Score**: {match['score']:.3f}  {extract_info(index_path)}")

# --- navigation ---
st.caption(f"Record {st.session_state.idx + 1} of {total}")

col_prev, col_next = st.columns(2)

with col_prev:
    if st.button('PREV', shortcut='Left') and st.session_state.idx > 0:
        st.session_state.idx -= 1

with col_next:
    if st.button('NEXT', shortcut='Right') and st.session_state.idx < total - 1:
        st.session_state.idx += 1