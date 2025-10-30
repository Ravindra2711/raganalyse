import json
import os
import requests
import streamlit as st


def backend_url_default() -> str:
    return os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


st.set_page_config(page_title="Auto-Analyst Frontend", page_icon="📊", layout="wide")
st.title("Auto-Analyst – Backend-Driven UI")

with st.sidebar:
    st.header("Backend")
    api_base = st.text_input("FastAPI base URL", value=backend_url_default())
    st.caption("Example: http://localhost:8000")


def api_post(path: str, *, headers=None, data=None, json_body=None, files=None):
    url = f"{api_base}{path}"
    return requests.post(url, headers=headers, data=data, json=json_body, files=files, allow_redirects=True)


def api_get(path: str, *, headers=None, params=None):
    url = f"{api_base}{path}"
    return requests.get(url, headers=headers, params=params, allow_redirects=True)


tab_session, tab_data, tab_chat, tab_code, tab_deep, tab_analytics, tab_templates, tab_feedback = st.tabs([
    "Session", "Data", "Chat", "Code", "Deep Analysis", "Analytics", "Templates", "Feedback",
])

with tab_session:
    st.subheader("Session")
    if "session_id" not in st.session_state:
        r = api_post("/generate-session")
        if r.ok:
            st.session_state.session_id = r.json()["session_id"]
        else:
            st.error(f"Failed to generate session: {r.status_code} {r.text}")

    cols_top = st.columns(4)
    with cols_top[0]:
        st.text_input("Session ID", st.session_state.get("session_id", ""), key="session_id_input")
    with cols_top[1]:
        if st.button("Regenerate Session"):
            r = api_post("/generate-session")
            if r.ok:
                st.session_state.session_id = r.json()["session_id"]
            else:
                st.error(f"Failed: {r.status_code} {r.text}")
    with cols_top[2]:
        if st.button("Get Session Info") and st.session_state.get("session_id"):
            r = api_get("/api/session-info", headers={"X-Session-ID": st.session_state.session_id})
            st.json(r.json() if r.ok else {"error": r.text})
    with cols_top[3]:
        if st.button("Reset Session to Default") and st.session_state.get("session_id"):
            r = api_post("/reset-session", headers={"X-Session-ID": st.session_state.session_id}, json_body={"preserveModelSettings": True})
            st.json(r.json() if r.ok else {"error": r.text})

with tab_data:
    st.subheader("Data Upload & Preview")
    uploaded_xlsx = st.file_uploader("Upload .xlsx", type=["xlsx"], key="upl_xlsx")
    if uploaded_xlsx and st.button("List Sheets"):
        files = {"file": (uploaded_xlsx.name, uploaded_xlsx.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        r = api_post("/api/excel-sheets", headers={"X-Session-ID": st.session_state.session_id}, files=files)
        if r.ok:
            st.session_state["sheets_available"] = r.json().get("sheets", [])
            st.success(f"Sheets: {', '.join(st.session_state['sheets_available'])}")
        else:
            st.error(f"Failed: {r.status_code} {r.text}")

    sel = []
    if st.session_state.get("sheets_available"):
        sel = st.multiselect("Select sheets to ingest (optional)", st.session_state["sheets_available"], key="sel_xlsx")

    if uploaded_xlsx and st.button("Upload Excel"):
        files = {"file": (uploaded_xlsx.name, uploaded_xlsx.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        data = {
            "name": uploaded_xlsx.name,
            "description": "Uploaded via Streamlit",
            "fill_nulls": "true",
            "convert_types": "true",
        }
        if sel:
            data["selected_sheets"] = json.dumps(sel)
        r = api_post("/upload_excel", headers={"X-Session-ID": st.session_state.session_id}, files=files, data=data)
        st.json(r.json() if r.ok else {"error": r.text})

    st.divider()
    st.caption("CSV quick preview (no session change)")
    uploaded_csv = st.file_uploader("Preview .csv", type=["csv"], key="upl_csv")
    if uploaded_csv and st.button("Preview CSV"):
        files = {"file": (uploaded_csv.name, uploaded_csv.getvalue(), "text/csv")}
        r = api_post("/preview-csv-upload", files=files)
        st.json(r.json() if r.ok else {"error": r.text})

    st.divider()
    st.caption("Set model settings for this session")
    provider = st.selectbox("Provider", ["openai", "anthropic", "groq", "gemini"], index=3)
    model = st.text_input("Model", value="gemini/gemini-2.5-flash")
    temperature = st.number_input("Temperature", value=0.1, min_value=0.0, max_value=1.0, step=0.1)
    max_tokens = st.number_input("Max tokens", value=800, min_value=1, step=100)
    if st.button("Apply Model Settings"):
        payload = {"provider": provider, "model": model, "temperature": float(temperature), "max_tokens": int(max_tokens)}
        r = api_post("/settings/model", headers={"X-Session-ID": st.session_state.session_id, "Content-Type": "application/json"}, json_body=payload)
        st.json(r.json() if r.ok else {"error": r.text})
    if st.button("Current Model Settings"):
        r = api_get("/api/model-settings", headers={"X-Session-ID": st.session_state.session_id})
        st.json(r.json() if r.ok else {"error": r.text})

with tab_chat:
    st.subheader("Chat")
    listcol, createcol = st.columns(2)
    with createcol:
        if st.button("Create Chat"):
            r = api_post("/chats/", json_body={"user_id": 1})
            if r.ok:
                st.session_state.chat_id = r.json().get("chat_id")
            else:
                st.error(f"Create chat failed: {r.status_code} {r.text}")
    with listcol:
        if st.button("List Chats"):
            r = api_get("/chats")
            st.json(r.json() if r.ok else {"error": r.text})

    st.write(f"Current Chat ID: {st.session_state.get('chat_id', '(none)')}")
    q = st.text_input("Ask a question about the dataset")
    if q and st.session_state.get("chat_id") and st.button("Send"):
        payload = {
            "content": q,
            "model_name": "gemini/gemini-2.5-flash",
            "temperature": 0.1,
            "max_tokens": 800,
        }
        r = api_post(
            f"/chats/{st.session_state.chat_id}/query",
            headers={"X-Session-ID": st.session_state.session_id, "Content-Type": "application/json"},
            json_body=payload,
        )
        if r.ok:
            st.success("Answer")
            st.write(r.json().get("content", ""))
        else:
            st.error(f"Query failed: {r.status_code} {r.text}")

    viewcol, delcol = st.columns(2)
    with viewcol:
        if st.session_state.get("chat_id") and st.button("Show Chat History"):
            r = api_get(f"/chats/{st.session_state.chat_id}")
            st.json(r.json() if r.ok else {"error": r.text})
    with delcol:
        if st.session_state.get("chat_id") and st.button("Delete Chat"):
            r = requests.delete(f"{api_base}/chats/{st.session_state.chat_id}")
            st.json(r.json() if r.ok else {"error": r.text})

with tab_code:
    st.subheader("Code Execution")
    code_text = st.text_area("Python code to execute (sandboxed by backend)")
    if st.button("Execute Code"):
        payload = {"code": code_text}
        r = api_post("/code/execute", headers={"Content-Type": "application/json"}, json_body=payload)
        st.json(r.json() if r.ok else {"error": r.text})
    if st.button("Get Latest Code"):
        r = api_get("/code/get-latest-code")
        st.json(r.json() if r.ok else {"error": r.text})

with tab_deep:
    st.subheader("Deep Analysis (Streaming)")
    goal = st.text_input("Analysis goal")
    if st.button("Start Deep Analysis") and goal:
        try:
            resp = requests.post(f"{api_base}/deep_analysis_streaming", json={"goal": goal}, headers={"X-Session-ID": st.session_state.session_id}, stream=True)
            if resp.status_code != 200:
                st.error(f"Failed: {resp.status_code} {resp.text}")
            else:
                placeholder = st.empty()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    placeholder.write(line)
        except Exception as e:
            st.error(str(e))

with tab_analytics:
    st.subheader("Analytics")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Usage Summary"):
            r = api_get("/analytics/summary")
            st.json(r.json() if r.ok else {"error": r.text})
    with col_b:
        days = st.number_input("Days", value=7, min_value=1, max_value=60)
        if st.button("Model Usage (window)"):
            r = api_get("/analytics/model-usage", params={"days": int(days)})
            st.json(r.json() if r.ok else {"error": r.text})

with tab_templates:
    st.subheader("Templates & Preferences")
    st.caption("Refer to /templates endpoints in docs to manage templates.")

with tab_feedback:
    st.subheader("Feedback")
    msg_id = st.text_input("Message ID to rate")
    rating = st.slider("Rating", 1, 5, 5)
    if st.button("Submit Feedback") and msg_id:
        payload = {"message_id": int(msg_id), "rating": int(rating)}
        # If your backend requires a different route body, adjust accordingly
        r = api_post("/feedback/message/", headers={"Content-Type": "application/json"}, json_body=payload)
        st.json(r.json() if r.ok else {"error": r.text})

st.caption("Tip: Set BACKEND_URL env to point at a remote server. All actions require the backend running and reachable.")

