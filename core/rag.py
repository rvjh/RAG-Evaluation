from core.ingestion import IngestionPipeline
from core.retrieval import RetrievalPipeline
from core.generation import GenerationPipeline


class RAGPipeline:

    def __init__(self):
        self.ingestion = IngestionPipeline()
        self.retrieval = None
        self.generation = GenerationPipeline()

    def initialize(self):
        vectorstore = self.ingestion.run()
        self.retrieval = RetrievalPipeline(vectorstore)

    def retrieve(self, query):
        return self.retrieval.retrieve(query)

    def generate(self, query, docs):
        return self.generation.generate(query, docs)

    def run(self, query):
        docs = self.retrieve(query)
        answer = self.generate(query, docs)
        return answer, docs