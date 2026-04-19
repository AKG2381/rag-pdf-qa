import streamlit as st
import uuid
from datetime import datetime

from app.chain import ask, Response
from app.ingestor import ingest, clear_collection

# ---------------- STATE ---------------- #

def init_state():
    if "chats" not in st.session_state:
        st.session_state.chats = {}
    if "active" not in st.session_state:
        st.session_state.active = None

init_state()

# ---------------- CHAT MGMT ---------------- #

def create_chat():
    cid = str(uuid.uuid4())[:8]
    st.session_state.chats[cid] = {
        "title": f"Chat {len(st.session_state.chats)+1}",
        "messages": [],
        "files": [],
        "processing": False,
        "collection": cid,
        "created_at": datetime.now().strftime("%b %d, %H:%M")
    }
    st.session_state.active = cid

def get_chat():
    return st.session_state.chats.get(st.session_state.active)

# ---------------- SIDEBAR ---------------- #

with st.sidebar:
    if st.button("➕ New Chat"):
        create_chat()

    for cid, c in st.session_state.chats.items():
        if st.button(c["title"], key=cid):
            st.session_state.active = cid

# ---------------- MAIN ---------------- #

if not st.session_state.active:
    st.info("Create a new chat")
    st.stop()

c = get_chat()

# ---------------- TITLE ---------------- #

new_name = st.text_input(
    "Rename Chat",
    value=c["title"],
    key=f"rename_{st.session_state.active}",
    label_visibility="collapsed"
)

if new_name != c["title"]:
    c["title"] = new_name

# ---------------- FILE UPLOAD ---------------- #

uploads = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    accept_multiple_files=True,
    key=f"upload_{st.session_state.active}"
)

if uploads:
    for f in uploads:
        if f.name not in c["files"]:
            c["files"].append(f.name)
            c["processing"] = True

            with st.spinner(f"Processing {f.name}..."):
                try:
                    ingest(f, f.name, collection=c["collection"])
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

            c["processing"] = False

# ---------------- CHAT DISPLAY ---------------- #

for msg in c["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# ---------------- INPUT ---------------- #

disabled = c["processing"] or len(c["files"]) == 0

q = st.chat_input("Ask something...", disabled=disabled)

if q:
    c["messages"].append({"role": "user", "content": q})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp: Response = ask(q, collection=c["collection"])
                answer = resp.answer
            except Exception as e:
                answer = f"Error: {e}"

        st.write(answer)

    c["messages"].append({
        "role": "assistant",
        "content": answer
    })