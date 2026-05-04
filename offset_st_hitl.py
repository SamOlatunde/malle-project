'''
Module: hitl_ui.py

User interface for Human in the Loop Verification
'''
import json
import streamlit as st
from embed import stream_jsonl

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


@st.cache_data
def index_offsets(path: str) -> list[int]:
    '''Builds a list of byte offsets, one per line in the JSONL file.
    Cached by Streamlit so it only runs once per unique path.

    Args:
        path(str): path to the JSONL results file
    Returns:
        list[int]: byte offset of the start of each line
    '''
    offsets = []
    with open(path, 'rb') as f:
        while True:
            offsets.append(f.tell())
            line = f.readline()
            if not line:
                break
    return offsets[:-1]  # drop the trailing phantom offset at EOF


def fetch_record(path: str, offsets: list[int], idx: int) -> dict:
    '''Seeks directly to the record at idx and parses it.

    Args:
        path(str): path to the JSONL results file
        offsets(list[int]): byte offsets from index_offsets()
        idx(int): index of the record to fetch
    Returns:
        dict: parsed JSON record
    '''
    with open(path, 'rb') as f:
        f.seek(offsets[idx])
        return json.loads(f.readline())


# --- session state ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# --- build offset map (runs once, then cached) ---
offsets = index_offsets(results_path)
total = len(offsets)

# --- fetch current record ---
result = fetch_record(results_path, offsets, st.session_state.idx)
query_path = result['query_path']

# --- layout ---
col1, buff, col2 = st.columns([1,0.6, 2])

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