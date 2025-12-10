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

# Initialize searchers
sparse_searcher = LuceneSearcher('pyserini_indexes/owi_sample_lucineindex')

# Initialize CrossEncoder (MiniLM is fast and effective)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Load dataset
dataset = ir_datasets.load("owi/subsampled/dev")

# Build a mapping of query_id -> relevant doc_ids with relevance scores
qrels_dict = {}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

# Memory-efficient approach: Only load documents on-demand
# Option 1: Keep iterator available for on-demand loading
def get_document_text(doc_id):
    """Fetch document text on-demand by iterating through dataset"""
    for doc in dataset.docs_iter():
        if doc.doc_id == doc_id:
            text_parts = []
            if hasattr(doc, 'title') and doc.title:
                text_parts.append(doc.title)
            if hasattr(doc, 'description') and doc.description:
                text_parts.append(doc.description)
            if hasattr(doc, 'main_content') and doc.main_content:
                text_parts.append(doc.main_content)
            return " ".join(text_parts)
    return ""

# Option 2: Use Pyserini's document store directly (more efficient)
def get_document_from_index(doc_id, searcher):
    """Fetch document directly from Lucene index"""
    try:
        doc = searcher.doc(doc_id)
        if doc:
            return doc.raw()  # Returns the raw document content
        return ""
    except:
        return ""

def evaluate_searcher(searcher, searcher_name, k=1000, use_reranker=False, rerank_top_k=100, eval_at_k=[5, 10, 20, 50, 100]):
    """Evaluate a searcher on the dataset with optional cross-encoder reranking"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {searcher_name}")
    if use_reranker:
        print(f"  Using cross-encoder reranking on top-{rerank_top_k}")
    print('='*60)
    
    # Track metrics at different k values
    metrics_at_k = {k_val: {'precision': 0, 'recall': 0, 'ndcg': 0} for k_val in eval_at_k}
    total_mrr = 0
    num_queries = 0
    
    for query in dataset.queries_iter():
        query_id = query.query_id
        query_text = query.text
        
        # Skip if no qrels for this query
        if query_id not in qrels_dict:
            continue
        
        relevant_docs = qrels_dict[query_id]
        # Consider docs with relevance > 0 as relevant
        relevant_docids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
        
        if len(relevant_docids) == 0:
            continue
        
        # Initial retrieval
        try:
            hits = searcher.search(query_text, k=k)
            
            # Apply cross-encoder reranking if enabled
            if use_reranker and len(hits) > 0:
                # Take top rerank_top_k candidates for reranking
                candidates = hits[:rerank_top_k]
                
                # Prepare query-document pairs for cross-encoder
                pairs = []
                valid_hits = []
                for hit in candidates:
                    # Fetch document text from Lucene index (memory efficient)
                    doc_text = get_document_from_index(hit.docid, sparse_searcher)
                    if doc_text:
                        # Truncate to first 512 tokens to save memory and speed up inference
                        doc_text = ' '.join(doc_text.split()[:512])
                        pairs.append([query_text, doc_text])
                        valid_hits.append(hit)
                
                if len(pairs) > 0:
                    # Get cross-encoder scores
                    ce_scores = reranker.predict(pairs)
                    
                    # Sort by cross-encoder scores
                    scored_hits = list(zip(valid_hits, ce_scores))
                    scored_hits.sort(key=lambda x: x[1], reverse=True)
                    
                    # Reorder hits based on cross-encoder scores
                    reranked_hits = [hit for hit, score in scored_hits]
                    # Add back any hits that weren't reranked (beyond rerank_top_k)
                    remaining_hits = hits[rerank_top_k:]
                    hits = reranked_hits + remaining_hits
            
            retrieved_docids = [hit.docid for hit in hits]
        except Exception as e:
            print(f"Error searching for query {query_id}: {e}")
            continue
        
        # Calculate metrics at different k values
        for k_val in eval_at_k:
            retrieved_at_k = retrieved_docids[:k_val]
            relevant_retrieved_at_k = set(retrieved_at_k) & relevant_docids
            
            # Precision@K
            precision_at_k = len(relevant_retrieved_at_k) / len(retrieved_at_k) if retrieved_at_k else 0
            
            # Recall@K
            recall_at_k = len(relevant_retrieved_at_k) / len(relevant_docids) if relevant_docids else 0
            
            # NDCG@K
            dcg = 0
            for i, docid in enumerate(retrieved_at_k, 1):
                rel = relevant_docs.get(docid, 0)
                dcg += rel / np.log2(i + 1)
            
            # Ideal ranking (sorted by relevance)
            sorted_rels = sorted(relevant_docs.values(), reverse=True)[:k_val]
            idcg = 0
            for i, rel in enumerate(sorted_rels, 1):
                idcg += rel / np.log2(i + 1)
            
            ndcg_at_k = dcg / idcg if idcg > 0 else 0
            
            metrics_at_k[k_val]['precision'] += precision_at_k
            metrics_at_k[k_val]['recall'] += recall_at_k
            metrics_at_k[k_val]['ndcg'] += ndcg_at_k
        
        # MRR (only calculated once)
        rr = 0
        for i, docid in enumerate(retrieved_docids, 1):
            if docid in relevant_docids:
                rr = 1 / i
                break
        
        total_mrr += rr
        num_queries += 1
    
    # Print average metrics
    if num_queries > 0:
        print(f"\nResults over {num_queries} queries:")
        print(f"  MRR: {total_mrr/num_queries:.4f}")
        print()
        for k_val in eval_at_k:
            avg_p = metrics_at_k[k_val]['precision'] / num_queries
            avg_r = metrics_at_k[k_val]['recall'] / num_queries
            avg_n = metrics_at_k[k_val]['ndcg'] / num_queries
            print(f"  @{k_val:3d} -> P: {avg_p:.4f}  R: {avg_r:.4f}  NDCG: {avg_n:.4f}")
    else:
        print("No queries evaluated!")
    
    return {
        'name': searcher_name,
        'mrr': total_mrr/num_queries if num_queries > 0 else 0,
        **{f'p@{k_val}': metrics_at_k[k_val]['precision']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k},
        **{f'r@{k_val}': metrics_at_k[k_val]['recall']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k},
        **{f'ndcg@{k_val}': metrics_at_k[k_val]['ndcg']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k}
    }

# Evaluate different approaches
k = 1000
results = []

# BM25 only
bm25_results = evaluate_searcher(sparse_searcher, "BM25 (Lucene)", k=k, use_reranker=False)
results.append(bm25_results)

# BM25 + Cross-Encoder Reranking
bm25_rerank_results = evaluate_searcher(sparse_searcher, "BM25 + CrossEncoder", k=k, use_reranker=True, rerank_top_k=100)
results.append(bm25_rerank_results)

# Optional: Try different rerank_top_k values for BM25
for rerank_k in [10,25,50,75,100,200]:
    bm25_rerank = evaluate_searcher(
        sparse_searcher,
        f"BM25 + CE (top-{rerank_k})",
        k=k,
        use_reranker=True,
        rerank_top_k=rerank_k
    )
    results.append(bm25_rerank)

# Summary comparison
print(f"\n{'='*80}")
print("SUMMARY - Precision Comparison")
print('='*80)
print(f"{'Method':<25} | {'P@5':<7} | {'P@10':<7} | {'P@20':<7} | {'P@50':<7} | {'P@100':<7}")
print("-"*80)
for result in results:
    print(f"{result['name']:<25} | {result['p@5']:.4f}  | {result['p@10']:.4f}  | "
          f"{result['p@20']:.4f}  | {result['p@50']:.4f}  | {result['p@100']:.4f}")

print(f"\n{'='*80}")
print("SUMMARY - Recall & NDCG Comparison")
print('='*80)
print(f"{'Method':<25} | {'MRR':<7} | {'R@10':<7} | {'R@50':<7} | {'NDCG@10':<7} | {'NDCG@50':<7}")
print("-"*80)
for result in results:
    print(f"{result['name']:<25} | {result['mrr']:.4f}  | {result['r@10']:.4f}  | "
          f"{result['r@50']:.4f}  | {result['ndcg@10']:.4f}  | {result['ndcg@50']:.4f}")
