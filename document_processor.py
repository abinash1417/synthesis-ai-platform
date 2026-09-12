from pypdf import PdfReader
import chromadb
from config import Config


class KnowledgeBase:
    """Handles document ingestion, chunking, storage, and retrieval."""

    def __init__(self):
        self._client = chromadb.PersistentClient(path=Config.CHROMA_DB_PATH)
        self._collection = self._client.get_or_create_collection(name=Config.COLLECTION_NAME)

    @staticmethod
    def extract_pdf_text(file) -> str:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    @staticmethod
    def _chunk_text(text: str, max_chunk_size: int = Config.MAX_CHUNK_SIZE) -> list[str]:
        paragraphs = [
            p.strip() for p in text.split("\n")
            if len(p.strip()) > Config.MIN_CHUNK_LENGTH
        ]
        chunks = []
        for para in paragraphs:
            if len(para) <= max_chunk_size:
                chunks.append(para)
            else:
                for i in range(0, len(para), max_chunk_size):
                    chunks.append(para[i:i + max_chunk_size])
        return chunks

    def add_document(self, filename: str, text: str) -> int:
        chunks = self._chunk_text(text)
        existing_count = self._collection.count()

        ids = [f"{filename}_chunk_{existing_count + i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        self._collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, n_results: int = Config.SEARCH_RESULTS_COUNT) -> str:
        if self._collection.count() == 0:
            return "No documents have been uploaded yet."

        results = self._collection.query(query_texts=[query], n_results=n_results)
        chunks = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not chunks:
            return "No relevant information found in the documents."

        return "\n\n".join(
            f"[Source: {meta.get('source', 'unknown')}]\n{chunk}"
            for chunk, meta in zip(chunks, metadatas)
        )

    def get_full_document(self, filename: str) -> str:
        data = self._collection.get(where={"source": filename})
        if not data["documents"]:
            available = self.list_document_names()
            return f"No document found with filename '{filename}'. Available documents: {available}"
        return "\n".join(data["documents"])

    def list_document_names(self) -> list[str]:
        data = self._collection.get()
        return list(set(m.get("source", "unknown") for m in data["metadatas"]))

    def chunk_count(self) -> int:
        return self._collection.count()

    def clear(self):
        self._client.delete_collection(Config.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(name=Config.COLLECTION_NAME)