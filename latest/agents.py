# agents.py
"""
AutoGen + Gemini agents wrapper (finalized).
- Minimal Gemini config (gemini-2.5-flash)
- Robust reply normalization
- Guaranteed dataset context for all agents:
    * prefer index doc retrieved with retriever.retrieve("")
    * fallback to frontend dataset_summary stored in st.session_state
    * include df.head(5) CSV snippet when available
"""

import os
import json
import re
import traceback
from dataclasses import dataclass
from typing import Dict, Any, List

import streamlit as st

# autogen import
try:
    import autogen
    from autogen import ConversableAgent
except Exception as e:
    raise RuntimeError(
        "autogen import failed. Please install autogen-agentchat~=0.2 with gemini extras and google-generativeai.\n"
        "pip install 'autogen-agentchat[gemini]~=0.2' google-generativeai\n"
        f"Original error: {e}"
    )

# -------------------------
# Data classes for outputs
# -------------------------
@dataclass
class AgentResult:
    code: str
    commentary: str

@dataclass
class CodeCombinerResult:
    refined_complete_code: str

# -------------------------
# Gemini llm_config (minimal & strict)
# -------------------------
def make_gemini_llm_config():
    model = "gemini-2.5-flash"
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_GEMINI_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION"))

    if not api_key and not sa_path:
        raise EnvironmentError(
            "Gemini configuration missing: set GOOGLE_GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS. "
            "Also install google-generativeai."
        )

    cfg = {"model": model, "api_type": "google"}
    if api_key:
        cfg["api_key"] = api_key
    if sa_path:
        cfg["google_application_credentials"] = sa_path
    if project_id:
        cfg["project_id"] = project_id
    if location:
        cfg["location"] = location

    llm_config = {
        "config_list": [cfg],
        "temperature": 0.0,
        "max_tokens": 2000,
        "seed": 42,
    }
    return llm_config

# -------------------------
# Helpers for retrievers & parsing
# -------------------------
def safe_retrieve_first_text(retriever, query: str, fallback: str = "NO_DOCS_FOUND"):
    """
    Safely get first document text from a retriever.
    """
    try:
        docs = retriever.retrieve(query)
    except Exception as e:
        return f"RETRIEVER_EXCEPTION: {e}"

    if not docs:
        return fallback
    first = docs[0]
    if hasattr(first, "text"):
        return first.text or fallback
    if isinstance(first, dict) and "text" in first:
        return first.get("text") or fallback
    try:
        return str(first)
    except Exception:
        return fallback

def parse_json_like_reply(reply_text: Any) -> Dict[str, Any]:
    """
    Defensive parsing of model reply; accepts None/dict/str.
    Returns dict possibly containing 'code' and 'commentary' keys.
    """
    # Normalize
    if reply_text is None:
        text = ""
    elif isinstance(reply_text, dict):
        if "content" in reply_text:
            text = reply_text["content"] or ""
        elif "message" in reply_text and isinstance(reply_text["message"], dict):
            text = reply_text["message"].get("content", "") or ""
        else:
            try:
                text = json.dumps(reply_text)
            except Exception:
                text = str(reply_text)
    else:
        try:
            text = str(reply_text)
        except Exception:
            text = ""

    # JSON object extraction
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    except Exception:
        pass

    # Code block extraction
    code_blocks = re.findall(r"```(?:python)?\n([\s\S]*?)```", text, re.IGNORECASE)
    remainder = re.sub(r"```(?:python)?\n[\s\S]*?```", "", text, flags=re.IGNORECASE).strip()

    out: Dict[str, Any] = {}
    if code_blocks:
        out["code"] = "\n\n".join(code_blocks)
    if remainder:
        out["commentary"] = remainder
    if not out:
        out["commentary"] = text.strip()
    return out

# -------------------------
# LLM wrapper with robust run()
# -------------------------
class LLMAgentWrapper:
    def __init__(self, name: str, system_message: str):
        self.name = name
        llm_config = make_gemini_llm_config()
        try:
            self.agent = ConversableAgent(name=name, llm_config=llm_config)
        except TypeError:
            try:
                self.agent = ConversableAgent(name, llm_config)
            except Exception as e:
                raise RuntimeError(f"Failed to construct ConversableAgent: {e}")
        self.system_message = (system_message or "").strip()

    def run(self, messages: List[Dict[str, str]]):
        """
        Calls agent and returns a string (never None). Returns explicit markers for empty/exception.
        """
        msgs: List[Dict[str, str]] = []
        if self.system_message:
            msgs.append({"role": "user", "content": self.system_message})
        for m in messages:
            if isinstance(m, dict) and "content" in m:
                msgs.append(m)
            else:
                msgs.append({"role": "user", "content": str(m)})

        try:
            resp = None
            if hasattr(self.agent, "generate_reply"):
                resp = self.agent.generate_reply(messages=msgs)
            else:
                for fn in ("initiate_chat", "initiate_conversation", "chat"):
                    if hasattr(self.agent, fn):
                        func = getattr(self.agent, fn)
                        try:
                            resp = func(messages=msgs)
                        except TypeError:
                            try:
                                resp = func(message=msgs[-1]["content"])
                            except Exception:
                                resp = None
                        if resp is not None:
                            break
                if resp is None and callable(self.agent):
                    try:
                        resp = self.agent(msgs)
                    except Exception:
                        resp = None

            if resp is None:
                return "[[LLM_EMPTY_RESPONSE]]"
            if isinstance(resp, dict):
                if "content" in resp:
                    return "" if resp["content"] is None else str(resp["content"])
                if "message" in resp and isinstance(resp["message"], dict):
                    return "" if resp["message"].get("content") is None else str(resp["message"].get("content"))
                try:
                    return json.dumps(resp)
                except Exception:
                    return str(resp)
            if isinstance(resp, (str, bytes)):
                return resp.decode() if isinstance(resp, bytes) else resp
            return str(resp)
        except Exception as exc:
            # return a string (no exceptions bubbled)
            return f"[[LLM_EXCEPTION]] {type(exc).__name__}: {exc}\n{traceback.format_exc()}"

# -------------------------
# Agent prompt definitions
# -------------------------
PLANNER_PROMPT = (
    "You are the Data Analytics Planner. Inputs will be provided as:\n"
    "- Dataset: (string summary of dataset and columns)\n"
    "- Agent_desc: (string description of available agents)\n"
    "- Goal: the user's goal\n\n"
    "Output MUST be plain text in this exact format:\n\n"
    "plan: Agent1->Agent2->Agent3\n"
    "plan_desc: A short explanation of why that plan (each agent's role).\n\n"
    "Do not output anything else."
)

GOAL_REFINER_PROMPT = "Refine the user Goal into a single concise operational goal string suitable for the planner."

PREPROCESSING_PROMPT = (
    "You are the preprocessing agent. Input includes dataset summary and goal. "
    "Return ONLY a JSON object with keys: {\"code\": \"<python code>\", \"commentary\": \"<short explanation>\"}."
)
STATISTICAL_PROMPT = (
    "You are the statistical analytics agent. Input includes dataset summary and goal. "
    "Return ONLY a JSON object with keys: {\"code\": \"<python code>\", \"commentary\": \"<short explanation>\"}."
)
SKLEARN_PROMPT = (
    "You are the scikit-learn agent. Input includes dataset summary and goal. "
    "Return ONLY a JSON object with keys: {\"code\": \"<python code>\", \"commentary\": \"<short explanation>\"}."
)
DATA_VIZ_PROMPT = (
    "You are the visualization agent. Input includes dataset summary, styling hints and goal. "
    "Return ONLY a JSON object with keys: {\"code\": \"<python code>\", \"commentary\": \"<short explanation>\"}."
)
CODE_COMBINER_PROMPT = (
    "You are the code combiner. Input: dataset summary and a python list of agent code snippets. "
    "Return ONLY JSON: {\"refined_complete_code\": \"<full streamlit-ready python script>\"}."
)
MEMORY_SUMMARIZER_PROMPT = "You are the memory summarizer. Input: agent outputs and user goal. Return a short plain-text summary sentence."

AGENT_DEFINITIONS = {
    "analytical_planner": PLANNER_PROMPT,
    "goal_refiner_agent": GOAL_REFINER_PROMPT,
    "preprocessing_agent": PREPROCESSING_PROMPT,
    "statistical_analytics_agent": STATISTICAL_PROMPT,
    "sk_learn_agent": SKLEARN_PROMPT,
    "data_viz_agent": DATA_VIZ_PROMPT,
    "code_combiner_agent": CODE_COMBINER_PROMPT,
    "memory_summarize_agent": MEMORY_SUMMARIZER_PROMPT,
}

# -------------------------
# Dataset payload builder (guaranteed context)
# -------------------------
def build_dataset_payload(retriever, query: str) -> str:
    """
    Compose dataset text for prompts:
    1) try retriever.retrieve("") to get the index doc created by frontend (make_data doc)
    2) append frontend dataset_summary (st.session_state.dataset_summary) if present
    3) append df.head(5) CSV snippet for concrete rows/column names
    """
    # try to get index doc via empty query
    try:
        docs = retriever.retrieve("") if retriever is not None else []
    except Exception:
        docs = []

    dataset_doc_text = None
    if docs:
        first = docs[0]
        dataset_doc_text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else str(first))

    frontend_summary = st.session_state.get("dataset_summary")
    df_head_text = ""
    session_df = st.session_state.get("df")
    if session_df is not None:
        try:
            df_head_text = session_df.head(5).to_csv(index=False)
        except Exception:
            try:
                df_head_text = json.dumps(session_df.head(5).to_dict(orient="records"))
            except Exception:
                df_head_text = ""

    parts = []
    if dataset_doc_text:
        parts.append("[INDEX_DOC]\n" + dataset_doc_text)
    if frontend_summary:
        parts.append("[FRONTEND_SUMMARY]\n" + frontend_summary)
    if df_head_text:
        parts.append("[DF_HEAD_CSV]\n" + df_head_text)

    if not parts:
        return "No dataset summary or sample rows were provided."
    return "\n\n".join(parts)

# -------------------------
# Orchestrators
# -------------------------
class auto_analyst_ind:
    def __init__(self, agents: List[Any], retrievers: Dict[str, Any]):
        self.wrappers: Dict[str, LLMAgentWrapper] = {}
        self.agent_desc = []
        for a in agents:
            try:
                model_name = a.__pydantic_core_schema__['schema']['model_name']
            except Exception:
                model_name = str(a)
            prompt = AGENT_DEFINITIONS.get(model_name, "You are a helpful agent.")
            self.wrappers[model_name] = LLMAgentWrapper(model_name, prompt)
            self.agent_desc.append(model_name)

        # retriever-like objects expected
        self.dataset_retriever = retrievers['dataframe_index'].as_retriever(k=1)
        self.styling_retriever = retrievers['style_index'].as_retriever(similarity_top_k=1)
        self.memory_agent = LLMAgentWrapper("memory_summarize_agent", AGENT_DEFINITIONS["memory_summarize_agent"])

    def __call__(self, query: str, specified_agent: str):
        # Build guaranteed dataset context
        dataset_text = build_dataset_payload(self.dataset_retriever, query)
        style_text = safe_retrieve_first_text(self.styling_retriever, "", fallback="STYLE_NOT_FOUND")
        if style_text.startswith("STYLE_NOT_FOUND") or style_text.strip() == "":
            style_text = "No styling guidance provided."

        hint = st.session_state.get("st_memory", [])

        prompt_content = (
            f"Dataset:\n{dataset_text}\n\nStyling:\n{style_text}\n\n"
            f"Hint (short-term memory): {hint}\n\nGoal:\n{query}"
        )

        wrapper = self.wrappers.get(specified_agent)
        if wrapper is None:
            return {
                specified_agent: AgentResult(code=f"st.write('Agent {specified_agent} not found')", commentary=f"Agent {specified_agent} not found"),
                f"memory_{specified_agent}": "NO_MEMORY"
            }

        reply_text = wrapper.run([{"role": "user", "content": prompt_content}])
        parsed = parse_json_like_reply(reply_text)
        code = parsed.get("code", parsed.get("refined_complete_code", "")) or ""
        commentary = parsed.get("commentary", "") or ""

        mem_reply = self.memory_agent.run([{"role": "user", "content": f"agent_response:\n{reply_text}\n\nuser_goal:\n{query}"}])
        st.session_state.st_memory.insert(0, f"memory_{specified_agent} : {mem_reply}")
        st.session_state.st_memory = st.session_state.st_memory[:20]

        return {specified_agent: AgentResult(code=code, commentary=commentary), f"memory_{specified_agent}": mem_reply}

class auto_analyst:
    def __init__(self, agents: List[Any], retrievers: Dict[str, Any]):
        self.wrappers: Dict[str, LLMAgentWrapper] = {}
        self.agent_desc = []
        for a in agents:
            try:
                model_name = a.__pydantic_core_schema__['schema']['model_name']
            except Exception:
                model_name = str(a)
            prompt = AGENT_DEFINITIONS.get(model_name, "You are a helpful agent.")
            self.wrappers[model_name] = LLMAgentWrapper(model_name, prompt)
            self.agent_desc.append(model_name)

        self.planner = LLMAgentWrapper("analytical_planner", AGENT_DEFINITIONS["analytical_planner"])
        self.refiner = LLMAgentWrapper("goal_refiner_agent", AGENT_DEFINITIONS["goal_refiner_agent"])
        self.code_combiner = LLMAgentWrapper("code_combiner_agent", AGENT_DEFINITIONS["code_combiner_agent"])
        self.memory_agent = LLMAgentWrapper("memory_summarize_agent", AGENT_DEFINITIONS["memory_summarize_agent"])

        self.dataset_retriever = retrievers['dataframe_index'].as_retriever(k=1)
        self.styling_retriever = retrievers['style_index'].as_retriever(similarity_top_k=1)

    def __call__(self, query: str):
        dataset_text = build_dataset_payload(self.dataset_retriever, query)
        style_text = safe_retrieve_first_text(self.styling_retriever, "", fallback="STYLE_NOT_FOUND")
        if style_text.startswith("STYLE_NOT_FOUND") or style_text.strip() == "":
            style_text = "No styling guidance provided."

        hint = st.session_state.get("st_memory", [])
        agent_desc_str = str(self.agent_desc)

        planner_prompt = f"Dataset:\n{dataset_text}\n\nAgent_desc:\n{agent_desc_str}\n\nGoal:\n{query}"
        planner_reply = self.planner.run([{"role": "user", "content": planner_prompt}])

        plan_line = ""
        plan_desc_line = ""
        for ln in planner_reply.splitlines():
            if ln.strip().lower().startswith("plan:"):
                plan_line = ln.split(":", 1)[1].strip()
            if ln.strip().lower().startswith("plan_desc:"):
                plan_desc_line = ln.split(":", 1)[1].strip()

        if not plan_line:
            # try goal refiner once as fallback
            refined_goal_txt = self.refiner.run([{"role":"user","content":planner_prompt}]).strip()
            if refined_goal_txt:
                planner_reply = self.planner.run([{"role":"user","content":f"Dataset:\n{dataset_text}\n\nAgent_desc:\n{agent_desc_str}\n\nGoal:\n{refined_goal_txt}"}])
                for ln in planner_reply.splitlines():
                    if ln.strip().lower().startswith("plan:"):
                        plan_line = ln.split(":",1)[1].strip()
                    if ln.strip().lower().startswith("plan_desc:"):
                        plan_desc_line = ln.split(":",1)[1].strip()

        if not plan_line:
            plan_line = "preprocessing_agent"

        plan_list = [p.strip() for p in plan_line.split("->") if p.strip()]

        output: Dict[str, Any] = {}
        output["analytical_planner"] = {"plan": plan_line, "plan_desc": plan_desc_line}

        code_snippets: List[str] = []
        for agent_name in plan_list:
            wrapper = self.wrappers.get(agent_name)
            if wrapper is None:
                code = f"st.write('Agent {agent_name} not available')"
                commentary = f"Agent {agent_name} not found."
            else:
                prompt_content = (
                    f"Dataset:\n{dataset_text}\n\nStyling:\n{style_text}\n\nHint:\n{hint}\n\nGoal:\n{query}"
                )
                agent_reply = wrapper.run([{"role": "user", "content": prompt_content}])
                parsed = parse_json_like_reply(agent_reply)
                code = parsed.get("code", "")
                commentary = parsed.get("commentary", "")
            output[agent_name] = AgentResult(code=code, commentary=commentary)
            code_snippets.append(code)

        combiner_input = {"dataset": dataset_text, "agent_code_list": str(code_snippets)}
        combiner_reply = self.code_combiner.run([{"role": "user", "content": json.dumps(combiner_input)}])
        comb_parsed = parse_json_like_reply(combiner_reply)
        refined_code = comb_parsed.get("refined_complete_code", "\n\n".join(code_snippets))

        output["code_combiner_agent"] = CodeCombinerResult(refined_complete_code=refined_code)

        mem_reply = self.memory_agent.run([{"role":"user","content":f"code_combiner_agent:\n{refined_code}\n\nuser_goal:\n{query}"}])
        output["memory_combined"] = mem_reply
        st.session_state.st_memory.insert(0, f"memory_combined : {mem_reply}")
        st.session_state.st_memory = st.session_state.st_memory[:20]

        return output
