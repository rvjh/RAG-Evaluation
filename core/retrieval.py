from utils.utils import timed

class RetrievalPipeline:

    def __init__(self, vectorstore, k=3):
        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    @timed
    def retrieve(self, query):
        return self.retriever.invoke(query)