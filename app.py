import os
import streamlit as st
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

st.set_page_config(page_title="AI Candidate Screening Agent", page_icon="👤", layout="wide")
st.title("👤 AI Candidate Screening Agent")
st.caption("LangChain + LangGraph + OpenAI + Streamlit")

def get_openai_api_key():
    try:
        key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Add it under Streamlit Cloud → App settings → Secrets."
        )
    return key

@st.cache_resource
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=get_openai_api_key(),
    )

class State(TypedDict):
    application: str
    experience_level: str
    skill_match: str
    response: str

def categorize_experience(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Based on the following job application, categorize the candidate as "
        "'Entry-level', 'Mid-level', or 'Senior-level'. "
        "Respond with exactly one of these three values and nothing else.\n\n"
        "Application: {application}"
    )
    chain = prompt | get_llm()
    experience_level = chain.invoke({"application": state["application"]}).content.strip()
    return {"experience_level": experience_level}

def assess_skillset(state: State) -> State:
    prompt = ChatPromptTemplate.from_template(
        "Assess the candidate's skillset against the job requirement. "
        "The role is a Software Developer role requiring relevant software development skills. "
        "Respond with exactly 'Match' or 'No Match' and nothing else.\n\n"
        "Application: {application}"
    )
    chain = prompt | get_llm()
    skill_match = chain.invoke({"application": state["application"]}).content.strip()
    return {"skill_match": skill_match}

def schedule_hr_interview(state: State) -> State:
    return {"response": "Candidate has been shortlisted for an HR interview."}

def escalate_to_recruiter(state: State) -> State:
    return {"response": "Candidate has senior-level experience but doesn't match job skills."}

def reject_application(state: State) -> State:
    return {"response": "Candidate doesn't meet JD and has been rejected."}

def route_app(state: State) -> str:
    if state["skill_match"] == "Match":
        return "schedule_hr_interview"
    elif state["experience_level"] == "Senior-level":
        return "escalate_to_recruiter"
    return "reject_application"

@st.cache_resource
def build_workflow():
    workflow = StateGraph(State)
    workflow.add_node("categorize_experience", categorize_experience)
    workflow.add_node("assess_skillset", assess_skillset)
    workflow.add_node("schedule_hr_interview", schedule_hr_interview)
    workflow.add_node("escalate_to_recruiter", escalate_to_recruiter)
    workflow.add_node("reject_application", reject_application)
    workflow.add_edge(START, "categorize_experience")
    workflow.add_edge("categorize_experience", "assess_skillset")
    workflow.add_conditional_edges(
        "assess_skillset",
        route_app,
        {
            "schedule_hr_interview": "schedule_hr_interview",
            "escalate_to_recruiter": "escalate_to_recruiter",
            "reject_application": "reject_application",
        },
    )
    workflow.add_edge("schedule_hr_interview", END)
    workflow.add_edge("escalate_to_recruiter", END)
    workflow.add_edge("reject_application", END)
    return workflow.compile()

def run_candidate_screening(application: str):
    return build_workflow().invoke({"application": application})

with st.sidebar:
    st.header("Screening Workflow")
    st.write("1. Categorize experience")
    st.write("2. Assess skillset")
    st.write("3. Route candidate")
    st.divider()
    st.caption("API key is read securely from Streamlit Secrets.")

application = st.text_area(
    "Candidate Application / Resume",
    height=300,
    placeholder=(
        "Paste the candidate's job application or resume here...\n\n"
        "Example: I have 10 years of experience in Java, Spring Boot, "
        "Microservices, AWS and Kafka."
    ),
)

if st.button("🔍 Screen Candidate", type="primary", use_container_width=True):
    if not application.strip():
        st.warning("Please enter a candidate application first.")
    else:
        try:
            with st.spinner("AI agent is screening the candidate..."):
                result = run_candidate_screening(application)
            st.success("Screening completed successfully.")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Experience Level")
                st.info(result["experience_level"])
            with col2:
                st.subheader("Skill Match")
                if result["skill_match"] == "Match":
                    st.success(result["skill_match"])
                else:
                    st.warning(result["skill_match"])
            st.subheader("Final Recommendation")
            st.write(result["response"])
        except Exception as exc:
            st.error("Screening failed.")
            st.exception(exc)

st.divider()
st.caption("Built with Streamlit, LangChain, LangGraph and OpenAI.")
