from langchain_groq import ChatGroq
from config.config import Config
from utils.utils import timed


class GenerationPipeline:

    def __init__(self):
        self.llm = ChatGroq(
            model=Config.LLM_MODEL,
            api_key=Config.GROQ_API_KEY
        )

    @timed
    def generate(self, query, docs):

        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
{query}
"""

        return self.llm.invoke(prompt).content