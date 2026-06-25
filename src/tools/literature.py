import requests
import chromadb
from chromadb.utils import embedding_functions
from config.settings import PUBMED_BASE_URL
import hashlib

# ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_or_create_collection(
    name="pubmed_abstracts",
    embedding_function=embedding_fn
)

def fetch_pubmed_abstracts(query: str, max_results: int = 5) -> list[dict]:
    """Fetch real PubMed abstracts for a query."""
    try:
        # search
        search_resp = requests.get(
            f"{PUBMED_BASE_URL}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
            timeout=10
        )
        ids = search_resp.json()["esearchresult"]["idlist"]
        if not ids:
            return []

        # fetch abstracts
        fetch_resp = requests.get(
            f"{PUBMED_BASE_URL}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "rettype": "abstract", "retmode": "xml"},
            timeout=15
        )

        # parse XML manually (no lxml needed)
        import re
        xml = fetch_resp.text
        abstracts = re.findall(r'<AbstractText[^>]*>(.*?)</AbstractText>', xml, re.DOTALL)
        titles = re.findall(r'<ArticleTitle>(.*?)</ArticleTitle>', xml, re.DOTALL)

        results = []
        for i, uid in enumerate(ids):
            title = titles[i] if i < len(titles) else "No title"
            abstract = abstracts[i] if i < len(abstracts) else "No abstract available"
            # clean XML tags
            abstract = re.sub(r'<[^>]+>', '', abstract).strip()
            title = re.sub(r'<[^>]+>', '', title).strip()
            results.append({"id": uid, "title": title, "abstract": abstract})

        return results

    except Exception as e:
        return []

def store_in_chromadb(papers: list[dict]):
    """Store paper abstracts in ChromaDB if not already stored."""
    for paper in papers:
        doc_id = hashlib.md5(paper["id"].encode()).hexdigest()
        existing = collection.get(ids=[doc_id])
        if not existing["ids"]:
            collection.add(
                documents=[f"{paper['title']}\n\n{paper['abstract']}"],
                ids=[doc_id],
                metadatas=[{"pubmed_id": paper["id"], "title": paper["title"]}]
            )

def semantic_search(query: str, n_results: int = 3) -> list[str]:
    """Semantic search over stored abstracts."""
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count)
        )
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        return []

def search_literature(query: str, max_results: int = 5) -> str:
    """
    Full RAG pipeline:
    1. Fetch real abstracts from PubMed
    2. Store in ChromaDB
    3. Semantic search for most relevant content
    4. Return rich context for agent
    """
    try:
        # fetch and store new papers
        papers = fetch_pubmed_abstracts(query, max_results)
        if papers:
            store_in_chromadb(papers)

        # semantic search over all stored abstracts
        relevant_docs = semantic_search(query, n_results=3)

        if not relevant_docs:
            # fallback to title-only if no abstracts
            if papers:
                titles = [f"- {p['title']}" for p in papers]
                return f"Literature on '{query}':\n" + "\n".join(titles)
            return f"No literature found for: {query}"

        # format context
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            # truncate long abstracts
            truncated = doc[:600] + "..." if len(doc) > 600 else doc
            context_parts.append(f"[Paper {i+1}]\n{truncated}")

        return f"Relevant literature for '{query}':\n\n" + "\n\n".join(context_parts)

    except Exception as e:
        return f"Literature search error: {str(e)}"