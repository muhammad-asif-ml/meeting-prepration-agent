import streamlit as st
from workflow import meeting_workflow, MeetingState
from dotenv import load_dotenv
import time
import os

# Load environment variables
load_dotenv()

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Meeting Prep Agent",
    page_icon="🤖",
    layout="wide"
)

# ---------------- Title ----------------
st.markdown(
    """
    <div style="text-align: center; padding: 1rem;">
        <h1>🤖 Meeting Preparation Assistant</h1>
        <p style="font-size: 18px; color: #555;">
        Your AI-powered co-pilot for research, insights, and strategies before a big meeting.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------- Sidebar Inputs ----------------
with st.sidebar:
    st.header("📌 Meeting Details")
    company = st.text_input("🏢 Company Name", placeholder="OpenAI")
    objective = st.text_area("🎯 Meeting Objective", placeholder="Discuss potential collaboration opportunities.")
    attendees = st.text_input("👥 Attendees", placeholder="CTO, Manager, CEO, HR")
    duration = st.number_input("⏱️ Meeting Duration (minutes)", min_value=15, max_value=120, value=60)
    focus = st.text_area("🔍 Focus Areas", placeholder="AI strategy, partnerships, growth")
    start_button = st.button("🚀 Prepare Meeting", use_container_width=True)


# ---------------- Run Workflow ----------------
if start_button:
    st.markdown("### 🔄 Preparing Your Meeting Brief...")

    progress_text = st.empty()
    progress_bar = st.progress(0)

    steps = [
        "Analyzing company background...",
        "Researching industry insights...",
        "Formulating meeting strategy...",
        "Creating executive brief..."
    ]

    # simulate stepper
    for i, step in enumerate(steps, 1):
        progress_text.markdown(f"**Step {i}/{len(steps)}:** {step}")
        time.sleep(1.5)  # <-- simulate waiting for each agent
        progress_bar.progress(i / len(steps))

    # Initial state
    state = MeetingState(
        company=company,
        objective=objective,
        attendees=attendees,
        duration=duration,
        focus=focus
    )

    # Run workflow
    final_state = meeting_workflow.invoke(state)

    # ---------------- Display Outputs ----------------
    st.success("✅ Meeting Brief Ready!")

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📖 Company Context", expanded=True):
            st.markdown(final_state.get("context", "_No context generated._"))

        with st.expander("🌍 Industry Insights", expanded=True):
            st.markdown(final_state.get("industry", "_No industry insights generated._"))

    with col2:
        with st.expander("📝 Meeting Strategy", expanded=True):
            st.markdown(final_state.get("strategy", "_No strategy generated._"))

        with st.expander("📑 Executive Brief", expanded=True):
            st.markdown(final_state.get("brief", "_No brief generated._"))


# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 14px;">
        ⚡ Powered by <b>LangChain</b> + <b>LangGraph</b> + <b>Streamlit</b>
    </div>
    """,
    unsafe_allow_html=True
)
