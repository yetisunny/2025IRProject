import os
import time
import json
from collections import defaultdict

from pyserini.search.hybrid import HybridSearcher
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
import ir_datasets
import ir_datasets_owi

# Load data
ir_datasets_owi.register()

# Initialize searchers
print("Initializing searchers...")
init_start = time.time()
dense_searcher = FaissSearcher('colbert_encoded_docs/', 'castorini/tct_colbert-v2-hnp-msmarco')
sparse_searcher = LuceneSearcher('pyserini_indexes/owi_sample_lucineindex')
hybrid_searcher = HybridSearcher(dense_searcher, sparse_searcher)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
init_time = time.time() - init_start
print(f"Initialization completed in {init_time:.2f} seconds\n")

# Load test dataset (no qrels)
dataset = ir_datasets.load("owi/subsampled/test")

def get_document_from_index(doc_id, searcher):
    """Fetch document directly from Lucene index"""
    try:
        doc = searcher.doc(doc_id)
        if doc:
            return doc.raw()
        return ""
    except:
        return ""

def retrieve_and_export_top_k(searcher, searcher_name, top_k=3, use_reranker=False, rerank_top_k=100):
    """
    Retrieve top-k documents for each query and export to a file for manual evaluation
    """
    print(f"\n{'='*60}")
    print(f"Running: {searcher_name}")
    if use_reranker:
        print(f"  Using cross-encoder reranking on top-{rerank_top_k}")
    print('='*60)
    
    results = []
    total_queries = 0
    start_time = time.time()
    
    for query in dataset.queries_iter():
        query_id = query.query_id
        query_text = query.text
        
        try:
            # Initial retrieval
            hits = searcher.search(query_text, k=1000)
            
            # Apply cross-encoder reranking if enabled
            if use_reranker and len(hits) > 0:
                # Take top rerank_top_k candidates for reranking
                candidates = hits[:rerank_top_k]
                
                # Prepare query-document pairs for cross-encoder
                pairs = []
                valid_hits = []
                
                for hit in candidates:
                    doc_text = get_document_from_index(hit.docid, sparse_searcher)
                    if doc_text:
                        doc_text = ' '.join(doc_text.split()[:512])
                        pairs.append([query_text, doc_text])
                        valid_hits.append(hit)
                
                if len(pairs) > 0:
                    # Get cross-encoder scores
                    ce_scores = reranker.predict(pairs)
                    
                    # Sort by cross-encoder scores
                    scored_hits = list(zip(valid_hits, ce_scores))
                    scored_hits.sort(key=lambda x: x[1], reverse=True)
                    
                    # Reorder hits
                    reranked_hits = [hit for hit, score in scored_hits]
                    remaining_hits = hits[rerank_top_k:]
                    hits = reranked_hits + remaining_hits
            
            # Get top-k results
            top_hits = hits[:top_k]
            
            # Fetch full documents for top-k
            top_docs = []
            for rank, hit in enumerate(top_hits, 1):
                doc_text = get_document_from_index(hit.docid, sparse_searcher)
                top_docs.append({
                    'rank': rank,
                    'doc_id': hit.docid,
                    'score': float(hit.score),
                    'doc_text': doc_text
                })
            
            results.append({
                'query_id': query_id,
                'query_text': query_text,
                'top_documents': top_docs
            })
            
            total_queries += 1
            
            if total_queries % 10 == 0:
                print(f"  Processed {total_queries} queries...")
        
        except Exception as e:
            print(f"Error processing query {query_id}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted {total_queries} queries in {elapsed_time:.2f}s")
    
    # Export results to JSON file
    output_filename = f"{searcher_name.replace(' ', '_').replace('+', 'plus').lower()}_top{top_k}_results.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results exported to: {output_filename}")
    
    # Also create a human-readable text file
    text_filename = f"{searcher_name.replace(' ', '_').replace('+', 'plus').lower()}_top{top_k}_results.txt"
    with open(text_filename, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"Top-{top_k} Results for: {searcher_name}\n")
        f.write(f"{'='*80}\n\n")
        
        for result in results:
            f.write(f"Query ID: {result['query_id']}\n")
            f.write(f"Query: {result['query_text']}\n")
            f.write(f"{'-'*80}\n")
            
            for doc in result['top_documents']:
                f.write(f"\n[Rank {doc['rank']}] Doc ID: {doc['doc_id']} | Score: {doc['score']:.4f}\n")
                f.write(f"{doc['doc_text'][:500]}...\n")  # First 500 chars
                f.write(f"{'-'*80}\n")
            
            f.write(f"\n{'='*80}\n\n")
    
    print(f"Human-readable results exported to: {text_filename}")
    
    return results


# Choose which retrieval method to use
print("\nSelect retrieval method:")
print("1. BM25 only")
print("2. Dense (ColBERT) only")
print("3. Hybrid")
print("4. BM25 + CrossEncoder Reranking")
print("5. Dense + CrossEncoder Reranking")
print("6. Hybrid + CrossEncoder Reranking")
print("7. Run all methods")

# For now, let's run a few key methods
# You can modify this section to run specific methods

# Run BM25
print("\n" + "="*80)
print("Running BM25...")
print("="*80)
bm25_results = retrieve_and_export_top_k(sparse_searcher, "BM25", top_k=3, use_reranker=False)

# Run Hybrid + CrossEncoder (often the best performing)
print("\n" + "="*80)
print("Running Hybrid + CrossEncoder...")
print("="*80)
hybrid_rerank_results = retrieve_and_export_top_k(
    hybrid_searcher, 
    "Hybrid_CrossEncoder", 
    top_k=3, 
    use_reranker=True, 
    rerank_top_k=100
)

print("\n" + "="*80)
print("PROCESSING COMPLETE")
print("="*80)
print("\nOutput files generated:")
print("  - JSON files: for programmatic analysis")
print("  - TXT files: for human reading/annotation")
print("\nYou can now manually review the top-3 documents for each query!")
