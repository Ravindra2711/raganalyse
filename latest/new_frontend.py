# new_frontend.py
"""
Streamlit frontend for Auto-Analyst migrated to AutoGen + Gemini (finalized).
- Recreates original project's behavior: build a rich dataset summary (make_data if available),
  store it in the vector index doc, keep the real `df` in the Python runtime and inject it into
  exec() so LLM-generated code that references `df` runs correctly.
- Defensive: shows helpful tracebacks, debug toggles, retriever previews.
- Safe-ish exec: injects a controlled globals dict with common libs and df.
"""

import os
import json
import traceback
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Dict

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# local modules
import retrievers  # our retrievers factory and fallback
from agents import auto_analyst, auto_analyst_ind, AgentResult, CodeCombinerResult

# Try to import original retriever utilities (if repo had them originally)
# This may define make_data and initiatlize_retrievers/styling_instructions used by the original project.
try:
    import retrievers as retriever_module  # same name, but we expect make_data possibly present
    has_make_data = hasattr(retriever_module, "make_data")
    has_initialze_retrievers = hasattr(retriever_module, "initiatlize_retrievers")
    styling_instructions = getattr(retriever_module, "styling_instructions", None)
except Exception:
    retriever_module = None
    has_make_data = False
    has_initialze_retrievers = False
    styling_instructions = None

st.set_page_config(page_title="Auto-Analyst (AutoGen + Gemini) — Frontend", layout="wide")

# Session state setup
if "st_memory" not in st.session_state:
    st.session_state.st_memory = []
if "retrievers" not in st.session_state:
    st.session_state.retrievers = None
if "last_output" not in st.session_state:
    st.session_state.last_output = None
if "dataset_summary" not in st.session_state:
    st.session_state.dataset_summary = None
if "df" not in st.session_state:
    st.session_state.df = None

st.title("Auto-Analyst (AutoGen + Gemini)")
st.caption("Frontend — robust, debug-friendly. Make sure gemini auth is configured in env.")

# Sidebar controls and debug options
with st.sidebar:
    st.header("Configuration & Debug")
    st.markdown(
        "- Set authentication: `GOOGLE_GEMINI_API_KEY` or `GOOGLE_APPLICATION_CREDENTIALS`.\n"
        "- Install required packages:\n"
        "  `pip install \"autogen-agentchat[gemini]~=0.2\" google-generativeai streamlit pandas`\n"
    )
    debug_mode = st.checkbox("Enable debug output (show tracebacks + raw LLM replies)", value=False)
    show_retriever_preview = st.checkbox("Show retriever preview", value=False)
    show_dataset_summary_preview = st.checkbox("Show dataset-summary preview", value=True)
    st.markdown("---")
    st.write("Note: agents.py uses `gemini-2.5-flash` by default.")

# Helper functions
def extract_code_from_text(text: str) -> str:
    """
    If the model returned fenced code blocks, extract the first python block.
    Otherwise return text unchanged.
    """
    if not text:
        return ""
    # find triple-backtick blocks
    import re
    m = re.search(r"```(?:python)?\n([\s\S]*?)```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text

def safe_exec_generated_code(code: str, session_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Execute agent-generated code in a controlled globals dict.
    Returns dict with keys: success(bool), stdout(str), error(str or None)
    """
    # build minimal safe builtins (whitelist)
    safe_builtins = {
        "range": range,
        "len": len,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "print": print,
    }

    exec_globals = {
        "__builtins__": safe_builtins,
        # inject actual dataframe and common libs the generated code typically expects
        "df": session_df,
        "pd": pd,
        "np": np,
        "st": st,
        "plt": plt,
        "px": px,
    }

    out = {"success": False, "stdout": "", "error": None}
    try:
        compiled = compile(code, "<agent-generated-code>", "exec")
    except Exception as e:
        out["error"] = f"COMPILE_ERROR: {type(e).__name__}: {e}"
        return out

    buf = StringIO()
    try:
        with redirect_stdout(buf):
            exec(compiled, exec_globals, exec_globals)
        out["success"] = True
        out["stdout"] = buf.getvalue()
    except Exception as e:
        out["error"] = f"EXECUTION_ERROR: {type(e).__name__}: {e}\n" + traceback.format_exc()
    return out

# Layout columns
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Upload dataset (CSV)")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"Loaded dataset — {len(df)} rows x {len(df.columns)} cols")
            if st.checkbox("Show dataframe head"):
                st.dataframe(df.head(200))
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.session_state.df = None
    else:
        df = st.session_state.get("df")

    st.markdown("---")
    st.subheader("Build retrievers (index dataset summary into RAG store)")

    # Short description input (optional)
    dataset_description = st.text_input("Short dataset description (optional)", value="Uploaded dataset")

    if df is not None:
        if st.button("Build retrievers"):
            try:
                # Attempt to use original make_data() if present in retriever_module (keeps original doc structure)
                if has_make_data:
                    try:
                        dict_ = retriever_module.make_data(df, dataset_description)
                        doc = [str(dict_)]
                        dataset_summary_text = str(dict_)
                    except Exception as e:
                        # fallback to basic summary
                        st.warning(f"make_data failed: {e}. Falling back to basic summary.")
                        dataset_summary_text = (
                            f"DataFrame with {len(df)} rows and columns: {', '.join(list(df.columns[:10]))}. "
                            f"Sample rows: {df.head(3).to_dict(orient='records')}"
                        )
                        doc = [dataset_summary_text]
                else:
                    # fallback: lightweight summary
                    dataset_summary_text = (
                        f"DataFrame with {len(df)} rows and columns: {', '.join(list(df.columns[:10]))}. "
                        f"Sample rows: {df.head(3).to_dict(orient='records')}"
                    )
                    doc = [dataset_summary_text]

                # Use original initiatlize_retrievers if available (original project used VectorStoreIndex)
                try:
                    if has_initialze_retrievers:
                        retrs = retriever_module.initiatlize_retrievers(retriever_module.styling_instructions, doc)
                    else:
                        # fallback to simplified retriever factory from retrievers.py
                        retrs = retrievers.create_retrievers_from_csv(df=df, style_texts=[dataset_description])
                except Exception as e:
                    st.warning(f"initiatlize_retrievers failed or not available: {e}. Using fallback retriever.")
                    retrs = retrievers.create_retrievers_from_csv(df=df, style_texts=[dataset_description])

                # Save retriever and deterministic dataset summary to session_state
                st.session_state.retrievers = retrs
                st.session_state.dataset_summary = dataset_summary_text

                st.success("Retrievers built and dataset summary stored in session state.")
                if show_retriever_preview:
                    try:
                        d_preview = retrs["dataframe_index"].retrieve("")[:3]
                        st.write("Data retriever preview (first 3 docs):")
                        for i, doc in enumerate(d_preview):
                            text = getattr(doc, "text", doc if isinstance(doc, str) else str(doc))
                            st.write(f"- doc[{i}]: {text[:400]}")
                    except Exception as e:
                        st.write("Could not preview dataframe retriever:", e)
                if show_dataset_summary_preview:
                    st.markdown("**Dataset summary preview (first 800 chars):**")
                    st.write((dataset_summary_text[:800] + "...") if len(dataset_summary_text) > 800 else dataset_summary_text)
            except Exception as ex:
                st.error(f"Failed to build retrievers: {ex}")
                if debug_mode:
                    st.text(traceback.format_exc())
    else:
        st.info("Upload a CSV to enable retriever construction.")

    st.markdown("---")
    st.subheader("Query / Goal")
    user_query = st.text_area("Describe what you want to do with the data", height=160)
    specified_agent = st.selectbox("Single agent run (auto_analyst_ind)", options=["preprocessing_agent", "statistical_analytics_agent", "sk_learn_agent", "data_viz_agent"])

    run_pipeline = st.button("Run pipeline (auto_analyst)")
    run_single_agent = st.button("Run single agent (auto_analyst_ind)")

with col_right:
    st.subheader("Results & Execution")

    # Pipeline run
    if run_pipeline:
        if st.session_state.retrievers is None:
            st.error("No retrievers found. Upload CSV and click 'Build retrievers' first.")
        elif not user_query or not user_query.strip():
            st.error("Please enter a query or goal.")
        else:
            # prepare agent stubs
            class _Stub:
                def __init__(self, name):
                    self.__pydantic_core_schema__ = {'schema': {'model_name': name}, 'cls': name}

            agent_names = [
                "analytical_planner",
                "goal_refiner_agent",
                "preprocessing_agent",
                "statistical_analytics_agent",
                "sk_learn_agent",
                "data_viz_agent",
                "code_combiner_agent",
                "memory_summarize_agent",
            ]
            agent_stubs = [_Stub(n) for n in agent_names]

            try:
                aa = auto_analyst(agent_stubs, st.session_state.retrievers)
            except Exception as e:
                st.error("Failed to create auto_analyst instance.")
                if debug_mode:
                    st.text(traceback.format_exc())
                else:
                    st.write("Error:", e)
                aa = None

            if aa:
                st.info("Running auto_analyst pipeline — this may take some time (Gemini latency).")
                try:
                    out = aa(user_query)
                    st.session_state.last_output = out
                except Exception as e:
                    st.error("auto_analyst pipeline raised an exception.")
                    st.text(traceback.format_exc())
                    out = None

                if out:
                    # show dataset summary used (fallback) for visibility
                    ds_sum = st.session_state.get("dataset_summary")
                    if ds_sum:
                        st.markdown("**Dataset summary used (fallback / index doc):**")
                        st.write((ds_sum[:800] + "...") if len(ds_sum) > 800 else ds_sum)

                    # Planner
                    if "analytical_planner" in out:
                        st.markdown("### Planner")
                        st.write(out["analytical_planner"])

                    # Each agent output
                    for k, v in out.items():
                        if k in ("analytical_planner", "memory_combined", "code_combiner_agent"):
                            continue
                        st.markdown(f"### {k}")
                        try:
                            if isinstance(v, AgentResult):
                                st.write("Commentary:")
                                st.write(v.commentary or "_(none)_")
                                st.write("Code:")
                                st.code(v.code or "# No code produced", language="python")
                            else:
                                st.write(v)
                        except Exception:
                            st.write(v)

                    # Combined code
                    cca = out.get("code_combiner_agent")
                    combined_code_text = ""
                    if cca:
                        st.markdown("## Combined code (from code_combiner_agent)")
                        if isinstance(cca, CodeCombinerResult):
                            combined_code_text = cca.refined_complete_code or ""
                        else:
                            combined_code_text = str(cca)
                        st.code(combined_code_text or "# No combined code produced", language="python")
                        st.download_button("Download combined code (.py)", data=(combined_code_text or "").encode("utf-8"), file_name="combined_auto_analyst.py", mime="text/x-python")

                    # Optionally execute combined code (explicit user action)
                    if combined_code_text:
                        if st.checkbox("Execute combined code now (runs in this Streamlit process)", value=False):
                            # Extract actual code block if model returned fenced code
                            code_to_exec = extract_code_from_text(combined_code_text)
                            if not code_to_exec.strip():
                                st.error("No executable code found in combined code.")
                            else:
                                # Ensure df present
                                session_df = st.session_state.get("df")
                                if session_df is None:
                                    st.error("No DataFrame loaded in session_state.df — cannot execute code.")
                                else:
                                    exec_result = safe_exec_generated_code(code_to_exec, session_df)
                                    if exec_result["success"]:
                                        st.success("Execution completed.")
                                        if exec_result["stdout"]:
                                            st.subheader("Execution stdout")
                                            st.text(exec_result["stdout"])
                                    else:
                                        st.error("Execution failed.")
                                        st.text(exec_result["error"])

                    # Memory
                    if "memory_combined" in out:
                        st.markdown("### Memory (combined)")
                        st.write(out["memory_combined"])

                    if debug_mode:
                        st.markdown("#### Raw output (debug)")
                        try:
                            serial = {k: (v.__dict__ if hasattr(v, "__dict__") else str(v)) for k, v in out.items()}
                            st.json(serial)
                        except Exception:
                            st.write(out)

    # Single agent run
    if run_single_agent:
        if st.session_state.retrievers is None:
            st.error("No retrievers found. Upload CSV and click 'Build retrievers' first.")
        elif not user_query or not user_query.strip():
            st.error("Please enter a query or goal.")
        else:
            class _Stub:
                def __init__(self, name):
                    self.__pydantic_core_schema__ = {'schema': {'model_name': name}, 'cls': name}

            agent_names = [
                "preprocessing_agent",
                "statistical_analytics_agent",
                "sk_learn_agent",
                "data_viz_agent",
            ]
            agent_stubs = [_Stub(n) for n in agent_names]

            try:
                aa_ind = auto_analyst_ind(agent_stubs, st.session_state.retrievers)
            except Exception as e:
                st.error("Failed to create auto_analyst_ind instance.")
                if debug_mode:
                    st.text(traceback.format_exc())
                else:
                    st.write("Error:", e)
                aa_ind = None

            if aa_ind:
                st.info(f"Running single agent: {specified_agent}")
                try:
                    out = aa_ind(user_query, specified_agent)
                    st.session_state.last_output = out
                except Exception as e:
                    st.error("auto_analyst_ind raised an exception.")
                    st.text(traceback.format_exc())
                    out = None

                if out:
                    # display agent output
                    agent_res = out.get(specified_agent)
                    mem_key = f"memory_{specified_agent}"
                    st.markdown(f"### {specified_agent}")
                    if isinstance(agent_res, AgentResult):
                        st.write("Commentary:")
                        st.write(agent_res.commentary or "_(none)_")
                        st.write("Code:")
                        st.code(agent_res.code or "# No code produced", language="python")
                        st.download_button("Download agent code (.py)", data=(agent_res.code or "").encode("utf-8"), file_name=f"{specified_agent}.py", mime="text/x-python")
                    else:
                        st.write(agent_res)

                    if mem_key in out:
                        st.write("Memory summary:")
                        st.write(out[mem_key])

                    if debug_mode:
                        st.markdown("#### Raw single-agent output (debug)")
                        try:
                            st.json({k: str(v) for k, v in out.items()})
                        except Exception:
                            st.write(out)

    # Show keys/last run
    if st.session_state.last_output and st.checkbox("Show last run keys"):
        st.write("Last run keys:", list(st.session_state.last_output.keys()))
