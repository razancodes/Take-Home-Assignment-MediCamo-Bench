"""
Streamlit Audit Tool for Clinical LLM Micro-Benchmark.

Loads patients_sample_3.json + key_definitions.csv and provides:
1. Patient/visit selector with mandatory item highlights
2. Flat table view: raw fields side-by-side with structured fields
3. HbA1c timeline chart with out-of-range flags
4. Rule-based check flags from checks.py
5. Gemma-4-31B-it LLM triage suggestions (unverified)
6. Gold labeling support panel
7. Export to audit_notes.csv

Usage: streamlit run audit/app.py
"""

import csv
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parent  # one level above clinical-llm-benchmark/

PATIENTS_FILE = DATA_DIR / "patients_sample_3.json"
KEY_DEFS_FILE = DATA_DIR / "key_definitions.csv"
AUDIT_NOTES_FILE = SCRIPT_DIR / "audit_notes.csv"
ENV_FILE = PROJECT_DIR / ".env"
GOLD_DIR = PROJECT_DIR / "gold"
LLM_CACHE_FILE = SCRIPT_DIR / "llm_cache.json"

# Mandatory items (from assignment constraints)
MANDATORY_VISITS = {
    "P0006": "V00073",
    "P0003": "V00028",
    "P0010": "V00147",
}
MANDATORY_TESTS = {
    "P0006": ("hba1c", 23),
    "P0003": ("hba1c", 10),
    "P0010": ("hba1c", 5),
}

# ---------------------------------------------------------------------------
# Load environment
# ---------------------------------------------------------------------------
def _load_env():
    """Load .env file if exists."""
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())

_load_env()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_patients():
    """Load patient data from JSON."""
    if not PATIENTS_FILE.exists():
        st.error(f"Data file not found: {PATIENTS_FILE}")
        st.stop()
    with open(PATIENTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_key_definitions():
    """Load key definitions CSV."""
    if not KEY_DEFS_FILE.exists():
        return pd.DataFrame()
    # CSV has non-UTF-8 characters (byte 0x95) — use latin-1 fallback
    try:
        return pd.read_csv(KEY_DEFS_FILE, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(KEY_DEFS_FILE, encoding="latin-1")


def get_patient(patients, pid):
    """Get patient dict by ID."""
    for p in patients:
        if p["patient_id"] == pid:
            return p
    return None


def get_visit(patient, vid):
    """Get visit dict by ID."""
    for v in patient.get("visit_list", []):
        if v.get("visit_id") == vid:
            return v
    return None


def get_hba1c(patient):
    """Get HbA1c readings list for a patient."""
    for t in (patient.get("test_list") or []):
        if (t.get("test_name") or "").lower() == "hba1c":
            return t.get("readings", [])
    return []


# ---------------------------------------------------------------------------
# Checks integration
# ---------------------------------------------------------------------------
@st.cache_data
def run_checks(patients_json_str):
    """Run all rule-based checks (cached by data hash)."""
    patients = json.loads(patients_json_str)
    sys.path.insert(0, str(SCRIPT_DIR))
    from checks import run_all_checks
    return run_all_checks(patients)


# ---------------------------------------------------------------------------
FALLBACK_MODELS = [
    "google/gemma-4-31B-it",
    "Qwen/Qwen3.6-35B-A3B",
    "zai-org/GLM-5.2"
]

def get_hf_url(model_id: str) -> str:
    # Use the official serverless HF router endpoint; the model ID is passed in the JSON payload
    return "https://router.huggingface.co/v1/chat/completions"

def call_llm_triage(visit_excerpt: str) -> str | None:
    """
    Call LLMs with fallback to flag raw-vs-structured mismatches.
    """
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return None

    system_msg = (
        "You are a clinical data quality auditor. You are given a JSON excerpt "
        "from a patient visit record. Your job is to compare the raw free-text "
        "fields (raw_diagnosis, raw_complaints) against the structured/parsed "
        "fields (diagnoses, medications, complaints) and flag any mismatches, "
        "contradictions, missing mappings, or suspicious entries.\n\n"
        "Be specific: for each flag, cite the exact field name and value. "
        "Label each flag as HIGH/MODERATE/INFO severity.\n"
        "If everything looks consistent, say so."
    )

    last_error = None
    for model_id in FALLBACK_MODELS:
        url = get_hf_url(model_id)
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Audit this visit record:\n\n```json\n{visit_excerpt}\n```"},
            ],
            "max_tokens": 2048,
            "temperature": 0.3,
        }
        
        # Add special settings for Qwen to disable thinking mode
        if "Qwen" in model_id:
            payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            payload["temperature"] = 0.7
            
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            # Append model note so user knows which model answered
            content = f"*(Generated by `{model_id}`)*\n\n" + content
            return content
        except requests.exceptions.HTTPError as e:
            # If it's a server error (5xx) or rate limit (429), try next model
            if e.response.status_code >= 500 or e.response.status_code == 429:
                last_error = f"[ERROR] HTTP {e.response.status_code}: {e.response.text}"
                continue
            else:
                return f"[ERROR] HTTP {e.response.status_code}: {e.response.text}"
        except Exception as e:
            last_error = f"[ERROR] {e}"
            continue
            
    return f"All models failed. Last error: {last_error}"

def run_llm_bg(excerpt_json: str, llm_key: str, shared_dict: dict):
    """Background thread worker for LLM API calls."""
    try:
        response = call_llm_triage(excerpt_json)
        shared_dict[llm_key] = response
    except Exception as e:
        shared_dict[llm_key] = f"[ERROR] Background thread failed: {e}"

# ---------------------------------------------------------------------------
# Helper: build visit excerpt for display / LLM
# ---------------------------------------------------------------------------
def build_visit_excerpt(visit: dict) -> dict:
    """
    Build a trimmed visit excerpt (no case_sheet, no empty note fields).
    This mirrors what will be sent to models in Phase 3.
    """
    keep_keys = [
        "visit_id", "visit_date", "patient_age", "gender",
        "raw_diagnosis", "raw_complaints", "complaints", "diagnoses",
        "medications", "exit_next_date", "exit_next_visit_num",
    ]
    note_keys = [
        "diet_exercise", "tests_advised", "advice_note",
        "cns_note", "cvs_note", "ent_note",
        "other_examination_note", "pa_note", "quick_note",
    ]
    excerpt = {}
    for k in keep_keys:
        if k in visit:
            excerpt[k] = visit[k]
    for k in note_keys:
        val = visit.get(k)
        # Include only non-empty notes
        if val and val != [] and val != "" and val != [""]:
            excerpt[k] = val
    return excerpt


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical LLM Benchmark — Audit Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .mandatory-badge {
        background: #ff4b4b;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
    }
    .flag-high { background: #ff4b4b; color: white; padding: 2px 6px; border-radius: 3px; }
    .flag-moderate { background: #ffa500; color: white; padding: 2px 6px; border-radius: 3px; }
    .flag-info { background: #4b9ff4; color: white; padding: 2px 6px; border-radius: 3px; }
    .llm-flag { background: #9b59b6; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.85em; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05em; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
patients = load_patients()
key_defs = load_key_definitions()
patients_json_str = json.dumps(patients)
all_flags = run_checks(patients_json_str)

# Initialize robust background task states
if "bg_tasks" not in st.session_state:
    st.session_state.bg_tasks = set()
if "bg_results" not in st.session_state:
    st.session_state.bg_results = {}

# Initialize session state
if "llm_responses" not in st.session_state:
    st.session_state.llm_responses = {}
    if LLM_CACHE_FILE.exists():
        try:
            with open(LLM_CACHE_FILE, "r", encoding="utf-8") as f:
                disk_cache = json.load(f)
                # Only load valid responses, skip errors/empty
                for k, v in disk_cache.items():
                    if v and not v.startswith("[ERROR]"):
                        st.session_state.llm_responses[k] = v
        except Exception:
            pass

# Process any newly completed background tasks from the shared dict
just_finished = []
for k in list(st.session_state.bg_tasks):
    if k in st.session_state.bg_results:
        just_finished.append(k)

for k in just_finished:
    st.session_state.bg_tasks.remove(k)
    resp = st.session_state.bg_results.pop(k)
    
    if resp and not resp.startswith("[ERROR]"):
        # Success: Save to state and persist to disk
        st.session_state.llm_responses[k] = resp
        st.toast(f"✅ LLM Triage completed for {k.replace('_', ' ')}!", icon="🎉")
        try:
            with open(LLM_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.llm_responses, f, indent=2)
        except Exception:
            pass
    else:
        # Failure: Save to state so UI shows error, but DO NOT persist to disk
        err_msg = resp if resp else "[ERROR] Received empty response from API."
        st.session_state.llm_responses[k] = err_msg
        st.toast(f"❌ LLM Triage failed for {k.replace('_', ' ')}.", icon="🚨")

if "flag_decisions" not in st.session_state:
    st.session_state.flag_decisions = {}
if "gold_labels" not in st.session_state:
    st.session_state.gold_labels = {}
if "gold_notes" not in st.session_state:
    st.session_state.gold_notes = {}

# ---------------------------------------------------------------------------
# Sidebar: Patient & visit selector
# ---------------------------------------------------------------------------
st.sidebar.title("🔬 Clinical Audit Tool")
st.sidebar.markdown("---")

patient_ids = [p["patient_id"] for p in patients]
selected_pid = st.sidebar.selectbox("Patient", patient_ids, index=0)
patient = get_patient(patients, selected_pid)

visit_ids = [v["visit_id"] for v in patient["visit_list"]]
# Mark mandatory visits
display_vids = []
for vid in visit_ids:
    if MANDATORY_VISITS.get(selected_pid) == vid:
        display_vids.append(f"⚠️ {vid} [MANDATORY]")
    else:
        display_vids.append(vid)

selected_display = st.sidebar.selectbox("Visit", display_vids, index=0)
selected_vid = selected_display.replace("⚠️ ", "").replace(" [MANDATORY]", "")
visit = get_visit(patient, selected_vid)
is_mandatory = MANDATORY_VISITS.get(selected_pid) == selected_vid

# Sidebar stats
st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Overview")
for p in patients:
    pid = p["patient_id"]
    n_visits = len(p.get("visit_list", []))
    hba1c = get_hba1c(p)
    n_hba1c = len(hba1c)
    mandatory = "⚠️" if pid in MANDATORY_VISITS else ""
    st.sidebar.markdown(f"**{pid}** {mandatory} — {n_visits} visits, {n_hba1c} HbA1c")

# Flag summary in sidebar
pid_flags = [f for f in all_flags if f["patient_id"] == selected_pid or f["patient_id"] == "ALL"]
n_high = len([f for f in pid_flags if f["severity"] == "high"])
n_mod = len([f for f in pid_flags if f["severity"] == "moderate"])
n_info = len([f for f in pid_flags if f["severity"] == "info"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"### Flags for {selected_pid}")
st.sidebar.markdown(
    f'<span class="flag-high">{n_high} HIGH</span> '
    f'<span class="flag-moderate">{n_mod} MOD</span> '
    f'<span class="flag-info">{n_info} INFO</span>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Main content — tabbed layout
# ---------------------------------------------------------------------------
st.title("Clinical LLM Micro-Benchmark — Data Audit")

if is_mandatory:
    st.markdown(
        '<span class="mandatory-badge">⚠️ MANDATORY EVALUATION ITEM</span>',
        unsafe_allow_html=True,
    )

tab_visit, tab_hba1c, tab_flags, tab_llm, tab_gold, tab_export = st.tabs([
    "📋 Visit View", "📈 HbA1c Timeline", "🚩 Automated Flags",
    "🤖 LLM Triage", "🏷️ Gold Labels", "💾 Export",
])

# ========================= TAB 1: Visit View =========================
with tab_visit:
    if visit is None:
        st.warning("Visit not found.")
    else:
        st.subheader(f"Visit {selected_vid} — {visit.get('visit_date', '?')} "
                     f"(Age {visit.get('patient_age', '?')}, {visit.get('gender', '?')})")

        col_raw, col_struct = st.columns(2)

        with col_raw:
            st.markdown("### 📝 Raw / Free-Text Fields")

            raw_dx = visit.get("raw_diagnosis")
            st.markdown("**raw_diagnosis:**")
            if isinstance(raw_dx, list):
                for i, item in enumerate(raw_dx):
                    st.markdown(f"  {i+1}. `{item}`")
            elif raw_dx:
                st.markdown(f"  `{raw_dx}`")
            else:
                st.markdown("  *(empty)*")

            raw_cx = visit.get("raw_complaints")
            st.markdown("**raw_complaints:**")
            if isinstance(raw_cx, list):
                for i, item in enumerate(raw_cx):
                    st.markdown(f"  {i+1}. `{item}`")
            elif raw_cx:
                st.markdown(f"  `{raw_cx}`")
            else:
                st.markdown("  *(empty)*")

            # Show non-empty notes
            st.markdown("**Notes:**")
            note_keys = ["advice_note", "cns_note", "cvs_note", "ent_note",
                         "other_examination_note", "pa_note", "quick_note"]
            any_note = False
            for nk in note_keys:
                val = visit.get(nk)
                if val and val != [] and val != "" and val != [""]:
                    st.markdown(f"  *{nk}:* `{val}`")
                    any_note = True
            if not any_note:
                st.markdown("  *(all empty)*")

        with col_struct:
            st.markdown("### 🏗️ Structured / Parsed Fields")

            st.markdown("**diagnoses:**")
            for dx in (visit.get("diagnoses") or []):
                neg = "🚫" if dx.get("is_negated") else "✅"
                mapped = "✓" if dx.get("is_mapped") else "✗"
                st.markdown(f"  {neg} `{dx.get('diagnosis', '?')}` "
                            f"(mapped={mapped}, neg={dx.get('is_negated')})")

            st.markdown("**complaints:**")
            for cx in (visit.get("complaints") or []):
                st.markdown(f"  • `{cx}`")
            if not visit.get("complaints"):
                st.markdown("  *(empty)*")

            # Visit metadata
            st.markdown("**Scheduling:**")
            st.markdown(f"  exit_next_date: `{visit.get('exit_next_date', 'N/A')}`")
            st.markdown(f"  exit_next_visit_num: `{visit.get('exit_next_visit_num', 'N/A')}`")

        # Medications table
        st.markdown("---")
        st.markdown("### 💊 Medications")
        meds = visit.get("medications") or []
        if meds:
            med_rows = []
            for m in meds:
                status = m.get("medicine_status")
                if isinstance(status, list):
                    status = str(status)
                med_rows.append({
                    "rx_med_name": m.get("rx_med_name", ""),
                    "medicine_name": m.get("medicine_name", ""),
                    "generic_name": m.get("generic_name") if not isinstance(m.get("generic_name"), list) else str(m.get("generic_name")),
                    "brand_name": m.get("brand_name", ""),
                    "status": status or "",
                    "dose": m.get("rx_dose", ""),
                    "frequency": m.get("rx_frequency", ""),
                    "duration": m.get("rx_duration", ""),
                })
            med_df = pd.DataFrame(med_rows)
            st.dataframe(med_df, use_container_width=True, hide_index=True)
        else:
            st.info("No medications for this visit.")

        # Visit-level flags inline
        visit_flags = [f for f in all_flags
                       if (f["patient_id"] == selected_pid and f["item_id"] == selected_vid)
                       or f["patient_id"] == "ALL"]
        if visit_flags:
            st.markdown("---")
            st.markdown("### 🚩 Flags for This Visit")
            for f in visit_flags:
                sev = f["severity"]
                css = f"flag-{sev}"
                st.markdown(
                    f'<span class="{css}">{sev.upper()}</span> '
                    f'**{f["check_name"]}** — {f["detail"]}',
                    unsafe_allow_html=True,
                )


# ========================= TAB 2: HbA1c Timeline =========================
with tab_hba1c:
    st.subheader(f"HbA1c Timeline — {selected_pid}")
    readings = get_hba1c(patient)

    if not readings:
        st.warning("No HbA1c readings found for this patient.")
    else:
        dates = []
        values = []
        for r in readings:
            try:
                d = datetime.strptime(r["date"], "%Y-%m-%d")
                v = float(r["value"])
                dates.append(d)
                values.append(v)
            except (ValueError, KeyError):
                continue

        fig = go.Figure()

        # Target zone shading (6.0–7.0% for diabetic patients)
        fig.add_hrect(
            y0=6.0, y1=7.0,
            fillcolor="rgba(0, 200, 0, 0.1)",
            line_width=0,
            annotation_text="Target (6–7%)",
            annotation_position="top left",
        )

        # Warning zone
        fig.add_hrect(
            y0=8.0, y1=max(values) + 1,
            fillcolor="rgba(255, 0, 0, 0.05)",
            line_width=0,
        )

        # Main line
        fig.add_trace(go.Scatter(
            x=dates, y=values,
            mode="lines+markers",
            name="HbA1c (%)",
            line=dict(color="#4b9ff4", width=2),
            marker=dict(size=8),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra></extra>",
        ))

        # Flag readings > 8% in red
        high_dates = [d for d, v in zip(dates, values) if v > 8.0]
        high_vals = [v for v in values if v > 8.0]
        if high_dates:
            fig.add_trace(go.Scatter(
                x=high_dates, y=high_vals,
                mode="markers",
                name="Above 8%",
                marker=dict(color="red", size=12, symbol="diamond"),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}%<extra>⚠️ High</extra>",
            ))

        # Annotate gaps > 6 months
        for i in range(1, len(dates)):
            gap_days = (dates[i] - dates[i - 1]).days
            if gap_days > 180:
                mid_date = dates[i - 1] + (dates[i] - dates[i - 1]) / 2
                mid_val = (values[i - 1] + values[i]) / 2
                fig.add_annotation(
                    x=mid_date, y=mid_val,
                    text=f"⚠️ {gap_days}d gap",
                    showarrow=True,
                    arrowhead=2,
                    bgcolor="rgba(255, 165, 0, 0.8)",
                    font=dict(color="white", size=10),
                )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="HbA1c (%)",
            hovermode="x unified",
            height=450,
            margin=dict(l=50, r=20, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Readings table
        st.markdown("### All Readings")
        readings_df = pd.DataFrame([
            {
                "Date": r.get("date", ""),
                "Value (%)": r.get("value", ""),
                "Units field": r.get("units", r.get("unit", "")),
            }
            for r in readings
        ])
        st.dataframe(readings_df, use_container_width=True, hide_index=True)

        # HbA1c-specific flags
        hba1c_flags = [f for f in all_flags
                       if f["patient_id"] == selected_pid and f["item_id"] == "hba1c"]
        if hba1c_flags:
            st.markdown("### 🚩 HbA1c Flags")
            for f in hba1c_flags:
                sev = f["severity"]
                css = f"flag-{sev}"
                st.markdown(
                    f'<span class="{css}">{sev.upper()}</span> '
                    f'**{f["check_name"]}** — {f["detail"]}',
                    unsafe_allow_html=True,
                )


# ========================= TAB 3: Automated Flags =========================
with tab_flags:
    st.subheader(f"All Automated Flags — {selected_pid}")

    # Show global flags too
    patient_flags = [f for f in all_flags
                     if f["patient_id"] == selected_pid or f["patient_id"] == "ALL"]

    if not patient_flags:
        st.success("No flags for this patient!")
    else:
        # Sort by severity
        sev_order = {"high": 0, "moderate": 1, "info": 2}
        patient_flags.sort(key=lambda f: sev_order.get(f["severity"], 3))

        for i, f in enumerate(patient_flags):
            sev = f["severity"]
            css = f"flag-{sev}"
            flag_key = f"{f['patient_id']}_{f['item_id']}_{f['check_name']}_{i}"

            with st.container():
                cols = st.columns([0.7, 0.15, 0.15])
                with cols[0]:
                    st.markdown(
                        f'<span class="{css}">{sev.upper()}</span> '
                        f'**[{f["item_id"]}]** {f["check_name"]}: {f["detail"]}',
                        unsafe_allow_html=True,
                    )
                with cols[1]:
                    if st.button("✅ Accept", key=f"accept_{flag_key}"):
                        st.session_state.flag_decisions[flag_key] = "accepted"
                with cols[2]:
                    if st.button("❌ Reject", key=f"reject_{flag_key}"):
                        st.session_state.flag_decisions[flag_key] = "rejected"

                # Show current decision
                decision = st.session_state.flag_decisions.get(flag_key)
                if decision:
                    st.caption(f"Decision: {decision}")


# ========================= TAB 4: LLM Triage =========================
with tab_llm:
    st.subheader("🤖 LLM Triage — Unverified Suggestions")
    st.markdown(
        '<span class="llm-flag">⚠️ UNVERIFIED LLM FLAGS — '
        'These are machine suggestions, NOT ground truth. '
        'Accept/reject each one manually.</span>',
        unsafe_allow_html=True,
    )

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        st.warning(
            "No HF_TOKEN found. Set it in the `.env` file to enable LLM triage.\n\n"
            "The `.env` file should contain:\n```\nHF_TOKEN=hf_your_token_here\n```"
        )
    else:
        st.info(f"Models (with fallback): `{'` → `'.join(FALLBACK_MODELS)}`")

        if visit:
            excerpt = build_visit_excerpt(visit)
            excerpt_json = json.dumps(excerpt, indent=2, default=str)

            with st.expander("View visit excerpt to be sent to LLM"):
                st.code(excerpt_json, language="json")

            llm_key = f"{selected_pid}_{selected_vid}"

            # Determine if we should show 'Run' or 'Re-run' based on cache
            btn_text = "🔄 Re-run LLM Triage" if llm_key in st.session_state.llm_responses else "🔍 Run LLM Triage on This Visit"
            
            if llm_key in st.session_state.bg_tasks:
                st.info("⏳ LLM Triage is currently running in the background. You can safely switch to other visits while you wait...")
            else:
                if st.button(btn_text, type="primary"):
                    if llm_key in st.session_state.llm_responses:
                        del st.session_state.llm_responses[llm_key]
                    st.session_state.bg_tasks.add(llm_key)
                    threading.Thread(target=run_llm_bg, args=(excerpt_json, llm_key, st.session_state.bg_results), daemon=True).start()
                    st.toast(f"Started background LLM triage for {selected_vid}...", icon="⏳")
                    st.rerun()

            # Display previous response if exists and is not currently running
            if llm_key in st.session_state.llm_responses and llm_key not in st.session_state.bg_tasks:
                response = st.session_state.llm_responses[llm_key]
                st.markdown("---")
                st.markdown("### LLM Response")
                st.markdown(
                    '<span class="llm-flag">⚠️ UNVERIFIED — Review each point below</span>',
                    unsafe_allow_html=True,
                )
                if response and not response.startswith("[ERROR]"):
                    edited_response = st.text_area(
                        "Review and Edit LLM Findings:", 
                        value=response, 
                        height=400,
                        key=f"edit_llm_text_{llm_key}"
                    )
                    
                    # Accept/reject for the whole response
                    llm_decision_key = f"llm_{llm_key}"
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("💾 Save & Accept Findings", key=f"accept_llm_{llm_key}"):
                            st.session_state.llm_responses[llm_key] = edited_response
                            st.session_state.flag_decisions[llm_decision_key] = "accepted"
                            try:
                                with open(LLM_CACHE_FILE, "w", encoding="utf-8") as f:
                                    json.dump(st.session_state.llm_responses, f, indent=2)
                            except Exception:
                                pass
                            st.toast("Saved and accepted findings!", icon="✅")
                    with col2:
                        if st.button("❌ Reject LLM Findings", key=f"reject_llm_{llm_key}"):
                            st.session_state.flag_decisions[llm_decision_key] = "rejected"
                    decision = st.session_state.flag_decisions.get(llm_decision_key)
                    if decision:
                        st.caption(f"Decision: {decision}")
                else:
                    st.error(response or "No response received.")


# ========================= TAB 5: Gold Labels =========================
with tab_gold:
    st.subheader("🏷️ Gold Labeling Support")

    if is_mandatory:
        st.success("✅ This is a MANDATORY evaluation item — label it here.")
    else:
        st.info("This visit is not in the mandatory set. Labels here are optional bonus items.")

    # Visit-level gold labeling
    st.markdown("### Visit-Level: Grounded Fact Inventory")
    st.markdown(
        "Record every diagnosis, medication, dose, frequency, date, and device/procedure "
        "that is **actually present** in the raw text. Mark the source field."
    )

    gold_key = f"{selected_pid}_{selected_vid}"

    # Initialize gold label state for this item
    if gold_key not in st.session_state.gold_labels:
        st.session_state.gold_labels[gold_key] = []

    # Add new fact
    with st.expander("➕ Add a grounded fact", expanded=is_mandatory and not st.session_state.gold_labels.get(gold_key)):
        fact_cols = st.columns([0.2, 0.3, 0.3, 0.2])
        with fact_cols[0]:
            fact_type = st.selectbox("Fact Type", [
                "diagnosis", "medication", "dose", "frequency",
                "date", "device/procedure", "lab_value", "other"
            ], key=f"ft_{gold_key}")
        with fact_cols[1]:
            fact_value = st.text_input("Value", key=f"fv_{gold_key}")
        with fact_cols[2]:
            source_field = st.selectbox("Source Field", [
                "raw_diagnosis", "raw_complaints", "medications.rx_med_name",
                "diagnoses", "complaints", "advice_note", "quick_note",
                "other_examination_note", "exit_next_date", "exit_next_visit_num",
                "other"
            ], key=f"sf_{gold_key}")
        with fact_cols[3]:
            grounded = st.selectbox("Grounded?", [
                "supported", "contradicted", "fabricated", "ambiguous", "not_specified"
            ], key=f"gr_{gold_key}")

        if st.button("Add Fact", key=f"add_{gold_key}"):
            if fact_value:
                st.session_state.gold_labels[gold_key].append({
                    "fact_type": fact_type,
                    "fact_value": fact_value,
                    "source_field": source_field,
                    "grounded": grounded,
                })
                st.rerun()

    # Display current facts
    current_facts = st.session_state.gold_labels.get(gold_key, [])
    if current_facts:
        st.markdown("#### Current Facts")
        facts_df = pd.DataFrame(current_facts)
        st.dataframe(facts_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear all facts for this item", key=f"clear_{gold_key}"):
            st.session_state.gold_labels[gold_key] = []
            st.rerun()

    # HbA1c gold labeling
    st.markdown("---")
    st.markdown("### HbA1c Series: Gold Readings")

    readings = get_hba1c(patient)
    if readings:
        hba1c_gold_key = f"{selected_pid}_hba1c"

        if hba1c_gold_key not in st.session_state.gold_labels:
            # Pre-populate from data
            st.session_state.gold_labels[hba1c_gold_key] = [
                {
                    "date": r.get("date", ""),
                    "value": r.get("value", ""),
                    "unit": "%" if r.get("units", "") == "" else r.get("units", ""),
                    "flagged": False,
                    "exclude": False,
                }
                for r in readings
            ]

        hba1c_facts = st.session_state.gold_labels[hba1c_gold_key]
        hba1c_df = pd.DataFrame(hba1c_facts)
        edited_df = st.data_editor(
            hba1c_df,
            column_config={
                "date": st.column_config.TextColumn("Date", disabled=True),
                "value": st.column_config.NumberColumn("Value", disabled=True),
                "unit": st.column_config.TextColumn("Unit", disabled=True),
                "flagged": st.column_config.CheckboxColumn("Flagged?"),
                "exclude": st.column_config.CheckboxColumn("Exclude?"),
            },
            use_container_width=True,
            hide_index=True,
            key=f"hba1c_editor_{hba1c_gold_key}",
        )
        # Save edits back
        st.session_state.gold_labels[hba1c_gold_key] = edited_df.to_dict("records")
    else:
        st.info("No HbA1c readings for this patient.")

    # Labeling notes
    st.markdown("---")
    st.markdown("### Labeling Notes")
    notes_key = f"notes_{gold_key}"
    note = st.text_area(
        "Notes for this item (labeling rationale, ambiguities, etc.)",
        value=st.session_state.gold_notes.get(notes_key, ""),
        key=f"ta_{notes_key}",
    )
    st.session_state.gold_notes[notes_key] = note


# ========================= TAB 6: Export =========================
with tab_export:
    st.subheader("💾 Export Audit Data")

    st.markdown("### Export Audit Notes CSV")
    st.markdown(
        "Exports all rule-based flags with your accept/reject decisions to "
        f"`{AUDIT_NOTES_FILE}`"
    )

    if st.button("📥 Export audit_notes.csv", type="primary"):
        rows = []
        for i, f in enumerate(all_flags):
            flag_key = f"{f['patient_id']}_{f['item_id']}_{f['check_name']}_{i}"
            decision = st.session_state.flag_decisions.get(flag_key)
            rows.append({
                "patient_id": f["patient_id"],
                "item_id": f["item_id"],
                "field": f["field"],
                "check_name": f["check_name"],
                "flag_source": f["flag_source"],
                "severity": f["severity"],
                "detail": f["detail"],
                "accepted": decision or "",
                "note": "",
                "timestamp": datetime.now().isoformat(),
            })

        # Add LLM triage decisions
        for key, response in st.session_state.llm_responses.items():
            decision = st.session_state.flag_decisions.get(f"llm_{key}")
            rows.append({
                "patient_id": key.split("_")[0],
                "item_id": key.split("_")[1] if "_" in key else "",
                "field": "llm_triage",
                "check_name": "gemma_4_31b_triage",
                "flag_source": "llm",
                "severity": "unverified",
                "detail": (response or "")[:500],
                "accepted": decision or "",
                "note": "",
                "timestamp": datetime.now().isoformat(),
            })

        with open(AUDIT_NOTES_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "patient_id", "item_id", "field", "check_name",
                "flag_source", "severity", "detail", "accepted",
                "note", "timestamp",
            ])
            writer.writeheader()
            writer.writerows(rows)

        st.success(f"✅ Exported {len(rows)} rows to `{AUDIT_NOTES_FILE}`")

    st.markdown("---")
    st.markdown("### Export Gold Label Drafts")
    st.markdown(
        "Exports current gold label state (visit facts + HbA1c readings) "
        "to `gold/gold_labels_draft.json`"
    )

    if st.button("📥 Export gold_labels_draft.json"):
        GOLD_DIR.mkdir(parents=True, exist_ok=True)
        draft = {
            "gold_labels": st.session_state.gold_labels,
            "gold_notes": st.session_state.gold_notes,
            "exported_at": datetime.now().isoformat(),
        }
        draft_path = GOLD_DIR / "gold_labels_draft.json"
        with open(draft_path, "w", encoding="utf-8") as f:
            json.dump(draft, f, indent=2, default=str)
        st.success(f"✅ Exported to `{draft_path}`")

    st.markdown("---")
    st.markdown("### Key Definitions Reference")
    if not key_defs.empty:
        st.dataframe(key_defs, use_container_width=True, hide_index=True)
    else:
        st.info("key_definitions.csv not found.")

# ---------------------------------------------------------------------------
# Background task auto-refresh trigger
# ---------------------------------------------------------------------------
if st.session_state.bg_tasks:
    time.sleep(2)
    st.rerun()
