import networkx as nx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from tqdm import tqdm # For tracking progress

import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
import ir_datasets
import numpy as np
import ir_datasets_owi

# Load data
ir_datasets_owi.register()
# Load dataset
dataset = ir_datasets.load("owi/subsampled/dev")
def extract_and_normalize_links(doc, base_url=None):
    """
    Extracts, resolves, and normalizes all outgoing hyperlinks from an OWIDoc,
    with error handling for malformed URLs.
    """
    source_url = doc.url
    if not base_url:
        base_url = source_url
    
    # 1. HTML Parsing (Use main_content which contains <a> tags)
    soup = BeautifulSoup(doc.main_content, 'html.parser')
    
    # 2. Extract and Normalize
    outgoing_links = set()
    for link_tag in soup.find_all('a', href=True):
        raw_target_link = link_tag.get('href')
        
        try:
            # Resolve relative URLs
            absolute_link = urljoin(base_url, raw_target_link)
            
            # Clean and Normalize the URL
            parsed_url = urlparse(absolute_link)
            
            # Create a canonical link format: scheme://netloc/path
            canonical_link = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path.rstrip('/')}"
            
            # Filter for http/https links that are not self-links
            if canonical_link.startswith(('http://', 'https://')) and canonical_link != source_url:
                outgoing_links.add(canonical_link)
        
        except ValueError as e:
            # Skip this link if parsing fails (e.g., Invalid IPv6 URL)
            # print(f"Skipping malformed link in {source_url}: {raw_target_link} -> {e}") 
            # (Uncomment above line for debugging which link caused the failure)
            continue
            
    return source_url, list(outgoing_links)

def compute_pagerank_for_dataset(dataset_iter):
    """
    Builds the graph and computes PageRank scores for all documents in the iterator.
    Filters links to ONLY include those pointing to documents found within the dataset.
    """
    print("\n--- Starting PageRank Graph Construction ---")
    
    # 1. First Pass: Collect all nodes (all doc URLs) and edges (links)
    all_docs = {} # Map doc_id -> url
    all_links = {} # Map url -> list of outgoing URLs
    
    # Use the document iterator from your dataset
    for doc in tqdm(dataset_iter, desc="1/2 Parsing Documents"):
        # We use the document URL as the primary graph node ID
        doc_url = doc.url
        all_docs[doc.doc_id] = doc_url
        
        # Extract links
        source_url, target_urls = extract_and_normalize_links(doc)
        all_links[source_url] = target_urls
    
    all_node_urls = set(all_docs.values())
    
    # 2. Build the NetworkX Graph
    G = nx.DiGraph()
    G.add_nodes_from(all_node_urls)
    
    edges_added = 0
    # Add edges, filtering targets to only include URLs present in the dataset
    for source_url, target_urls in tqdm(all_links.items(), desc="2/2 Building Graph"):
        for target_url in target_urls:
            # Check if the target URL is one of the nodes in our dataset
            if target_url in all_node_urls:
                G.add_edge(source_url, target_url)
                edges_added += 1

    print(f"Graph Built: {G.number_of_nodes()} Nodes, {edges_added} Edges.")
    
    # 3. Compute PageRank
    # Use the doc_id as the key for easier lookup later
    if G.number_of_nodes() > 0:
        print("Computing PageRank...")
        # Use standard damping factor (0.85)
        pagerank_scores_by_url = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-06)
        print("PageRank computation complete.")
    else:
        print("Graph is empty, returning uniform scores.")
        pagerank_scores_by_url = {}

    # 4. Map back to doc_id for indexing
    final_pagerank_scores = {}
    for doc_id, url in all_docs.items():
        # Handle cases where the URL might not be in the graph (e.g., if it was filtered out)
        final_pagerank_scores[doc_id] = pagerank_scores_by_url.get(url, 0) 
        
    return final_pagerank_scores


# Run PageRank on the subsample (Need to re-instantiate the iterator)
# Note: Since the dataset is small, the entire process runs on a single CPU core.
pagerank_scores_dict = compute_pagerank_for_dataset(dataset.docs_iter())
print(f"Computed PageRank scores for {len(pagerank_scores_dict)} documents.")
