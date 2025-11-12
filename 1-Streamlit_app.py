# streamlit run 1-Streamlit_app.py
# python== 3.10

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Streamlit configuration
st.set_page_config(page_title="LangChain Search Chat", page_icon="🔎")
st.title("🔎 LangChain - Chat with Search")

# Sidebar for API Key input
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# Initialize LLM only if API key is provided
if api_key:
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b")
else:
    st.warning("⚠️ Please enter your GROQ API Key to continue.")
    st.stop()

# Initialize tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
search_tool = DuckDuckGoSearchRun(name="Search")
tools = [search_tool, arxiv_tool, wiki_tool]

# Load prompt from ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an assistant for question-answering tasks.Use llm model, tools to answer the question precisely."
    "If you don't know the answer, say that you don't know."),MessagesPlaceholder(variable_name="agent_scratchpad"),("user", "{input}")])

# Create agent
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}]

# Display chat history
for i in st.session_state.messages:
    st.chat_message(i["role"]).write(i["content"])

# Handle user input
user_input = st.chat_input(placeholder="Ask anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    response = agent_executor.invoke({"input": user_input})
    final_response = response["output"]

    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.chat_message("assistant").write(final_response)
