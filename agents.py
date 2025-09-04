
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

search = TavilySearchResults(k=3)


search_tool = Tool(
    name="Search",
    func = search.run,
    description="Search the web for company and industry information"
)