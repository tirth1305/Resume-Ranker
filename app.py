import os
import streamlit as st
from typing import List
import PyPDF2
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class ResumeScore(BaseModel):
    score: float = Field(description="Score between 0 and 100")
    explanation: str = Field(description="Brief explanation of the score")

def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0.7
    )

def rank_resume_against_jd(jd_text: str, resume_text: str, llm) -> str:
    parser = PydanticOutputParser(pydantic_object=ResumeScore)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert resume reviewer. Your task is to evaluate how well a resume matches a job description.
        Consider factors like:
        - Required skills and experience
        - Education and qualifications
        - Relevant projects and achievements
        - Overall fit for the role
        
        Provide a score between 0-100 and a brief explanation.
        {format_instructions}"""),
        ("human", """Job Description:
        {jd_text}
        
        Resume:
        {resume_text}
        
        Please evaluate the match between this resume and job description.""")
    ])

    chain = prompt | llm | parser

    result = chain.invoke({
        "jd_text": jd_text,
        "resume_text": resume_text,
        "format_instructions": parser.get_format_instructions()
    })

    return f"{result.score:.1f}/100 - {result.explanation}"


#  UI 
st.set_page_config(page_title="Resume Ranker")
st.title("Resume Ranker with Azure OpenAI")

st.markdown("Upload a **Job Description** and up to **10 Resumes** in PDF format.")

jd_file = st.file_uploader("Upload Job Description (PDF)", type="pdf")
resume_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if jd_file and resume_files:
    with st.spinner("Extracting Job Description..."):
        jd_text = extract_text_from_pdf(jd_file)

    llm = get_llm()
    results = []

    for file in resume_files:
        with st.spinner(f"Ranking {file.name}..."):
            resume_text = extract_text_from_pdf(file)
            score = rank_resume_against_jd(jd_text, resume_text, llm)
            results.append((file.name, score))

    st.subheader("Ranked Resumes")
    for name, score in sorted(results, key=lambda x: float(x[1].split('/')[0]), reverse=True):
        st.write(f"**{name}**: {score}")
