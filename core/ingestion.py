import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config.config import Config
from utils.utils import timed


class IngestionPipeline:
    def __init__(self) -> None:
        self.embedding = HuggingFaceEmbeddings(
            model_name = Config.EMBEDDING_MODEL)
        
        self.vectorstore = None
    
    def load_documents(self):
        docs = []
        for file in os.listdir(Config.DATA_PATH):
            if file.endswith(".txt"):
                path = os.path.join(Config.DATA_PATH, file)

                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                docs.append(Document(page_content=text))

        return docs

    @timed
    def run(self):

        docs = self.load_documents()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(docs)

        self.vectorstore = Chroma.from_documents(
            chunks,
            embedding=self.embedding,
            persist_directory=Config.VECTOR_DB_PATH
        )

        return self.vectorstore