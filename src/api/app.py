import os

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Handbook Assistant", page_icon="📘")
st.title("📘 Enterprise Handbook Assistant")
st.caption("Answers come only from the handbook documents, with sources.")


@st.cache_data(ttl=300)
def get_departments() -> list[str]:
    return httpx.get(f"{API_URL}/departments", timeout=10).json()


with st.sidebar:
    st.header("Filters")
    try:
        chosen = st.multiselect("Departments (empty = all)", get_departments())
    except httpx.HTTPError:
        st.error("API not reachable. Start it with uvicorn.")
        st.stop()
    top_k = st.slider("Passages to use", 1, 10, 5)

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**[{s['n']}]** {s['title']} › {s['section']}  \n"
                        f"`{s['department']}/{s['source']}` · score {s['score']}"
                    )

if question := st.chat_input("Ask about the handbook..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and generating..."):
            try:
                r = httpx.post(
                    f"{API_URL}/ask",
                    json={"question": question, "departments": chosen or None, "top_k": top_k},
                    timeout=120,
                )
                r.raise_for_status()
                data = r.json()
            except httpx.HTTPError as e:
                data = {"answer": f"Request failed: {e}", "sources": []}
        st.markdown(data["answer"])
        with st.expander("Sources"):
            for s in data["sources"]:
                st.markdown(
                    f"**[{s['n']}]** {s['title']} › {s['section']}  \n"
                    f"`{s['department']}/{s['source']}` · score {s['score']}"
                )
    st.session_state.history.append(
        {"role": "assistant", "content": data["answer"], "sources": data["sources"]}
    )
