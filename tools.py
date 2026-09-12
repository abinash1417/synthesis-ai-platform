from langchain_core.tools import tool
from document_processor import KnowledgeBase

_kb = KnowledgeBase()


def get_knowledge_base() -> KnowledgeBase:
    return _kb


@tool
def search_documents(query: str) -> str:
    """Search the knowledge base for specific facts relevant to a query.
    Use this for targeted questions about specific details within documents."""
    return _kb.search(query)


@tool
def get_full_document(filename: str) -> str:
    """Retrieve the ENTIRE content of a specific document by filename.
    Use this when asked to review, analyze, summarize, or evaluate a whole document
    (e.g., 'how strong is this resume', 'summarize this report')."""
    return _kb.get_full_document(filename)