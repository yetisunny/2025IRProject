import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
import ir_datasets
import numpy as np
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.hybrid import HybridSearcher
import ir_datasets
import numpy as np
import ir_datasets_owi

#load data
ir_datasets_owi.register()
dataset = ir_datasets.load("owi/dev")
dataset = ir_datasets.load("owi/test")
dataset = ir_datasets.load("owi/subsampled/dev")
dataset = ir_datasets.load("owi/subsampled/test")
dataset = ir_datasets.load("owi/subsampled/dev")
# Initialize searchers
dense_searcher = FaissSearcher('colbert_encoded_docs/', 'castorini/tct_colbert-v2-hnp-msmarco')
sparse_searcher = LuceneSearcher('pyserini_indexes/owi_sample_lucineindex')

# Load dataset
dataset = ir_datasets.load("owi/subsampled/dev")

# Build a mapping of query_id -> relevant doc_ids with relevance scores
qrels_dict = {}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

def manual_hybrid_search(sparse_searcher, dense_searcher, query_text, k=1000, alpha=0.5):
    """Manual hybrid search with proper fusion"""
    # Get results from both
    sparse_hits = sparse_searcher.search(query_text, k=k)
    dense_hits = dense_searcher.search(query_text, k=k)
    
    # Normalize scores to [0,1]
    def normalize_scores(hits):
        if not hits:
            return {}
        scores = [hit.score for hit in hits]
        min_score, max_score = min(scores), max(scores)
        score_range = max_score - min_score if max_score > min_score else 1
        return {hit.docid: (hit.score - min_score) / score_range for hit in hits}
    
    sparse_scores = normalize_scores(sparse_hits)
    dense_scores = normalize_scores(dense_hits)
    
    # Combine scores: hybrid_score = alpha * dense + (1-alpha) * sparse
    all_docids = set(sparse_scores.keys()) | set(dense_scores.keys())
    
    hybrid_scores = {}
    for docid in all_docids:
        sparse_score = sparse_scores.get(docid, 0)
        dense_score = dense_scores.get(docid, 0)
        hybrid_scores[docid] = alpha * dense_score + (1 - alpha) * sparse_score
    
    # Sort and return top k
    sorted_docs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    return [docid for docid, score in sorted_docs]

def evaluate_searcher(searcher, searcher_name, k=1000, alpha=None):
    """Evaluate a searcher on the dataset"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {searcher_name}")
    print('='*60)
    
    total_precision = 0
    total_recall = 0
    total_mrr = 0
    total_ndcg = 0
    num_queries = 0
    
    for query in dataset.queries_iter():
        query_id = query.query_id
        query_text = query.text
        
        # Skip if no qrels for this query
        if query_id not in qrels_dict:
            continue
        
        relevant_docs = qrels_dict[query_id]
        # Consider docs with relevance > 0 as relevant (adjust as needed)
        relevant_docids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
        
        if len(relevant_docids) == 0:
            continue
        
        # Search
        try:
            if alpha is not None:
                # For hybrid searcher, pass alpha
                hits = searcher.search(query_text, k=k, alpha=alpha)
            else:
                hits = searcher.search(query_text, k=k)
            retrieved_docids = [hit.docid for hit in hits]
            print(len(retrieved_docids))
        except Exception as e:
            print(f"Error searching for query {query_id}: {e}")
            continue
        
        # Calculate metrics
        relevant_retrieved = set(retrieved_docids) & relevant_docids
        
        # Precision@K
        precision = len(relevant_retrieved) / len(retrieved_docids) if retrieved_docids else 0
        
        # Recall@K
        recall = len(relevant_retrieved) / len(relevant_docids) if relevant_docids else 0
        
        # MRR
        rr = 0
        for i, docid in enumerate(retrieved_docids, 1):
            if docid in relevant_docids:
                rr = 1 / i
                break
        
        # NDCG@K (simplified - using graded relevance)
        dcg = 0
        idcg = 0
        for i, docid in enumerate(retrieved_docids, 1):
            rel = relevant_docs.get(docid, 0)
            dcg += rel / np.log2(i + 1)
        
        # Ideal ranking (sorted by relevance)
        sorted_rels = sorted(relevant_docs.values(), reverse=True)[:k]
        for i, rel in enumerate(sorted_rels, 1):
            idcg += rel / np.log2(i + 1)
        
        ndcg = dcg / idcg if idcg > 0 else 0
        
        total_precision += precision
        total_recall += recall
        total_mrr += rr
        total_ndcg += ndcg
        num_queries += 1
    
    # Print average metrics
    if num_queries > 0:
        print(f"\nResults over {num_queries} queries:")
        print(f"  Precision@{k}: {total_precision/num_queries:.4f}")
        print(f"  Recall@{k}:    {total_recall/num_queries:.4f}")
        print(f"  MRR:           {total_mrr/num_queries:.4f}")
        print(f"  NDCG@{k}:      {total_ndcg/num_queries:.4f}")
    else:
        print("No queries evaluated!")
    
    return {
        'name': searcher_name,
        'precision': total_precision/num_queries if num_queries > 0 else 0,
        'recall': total_recall/num_queries if num_queries > 0 else 0,
        'mrr': total_mrr/num_queries if num_queries > 0 else 0,
        'ndcg': total_ndcg/num_queries if num_queries > 0 else 0
    }

# Evaluate different approaches
k = 1000
results = []

# BM25 only
bm25_results = evaluate_searcher(sparse_searcher, "BM25 (Lucene)", k=k)
results.append(bm25_results)

# Dense only
dense_results = evaluate_searcher(dense_searcher, "Dense (FAISS)", k=k)
results.append(dense_results)

# Hybrid with different alpha values
for alpha in [0.1, 0.25, 0.5, 0.75, 0.9]:
    hybrid_searcher = HybridSearcher(dense_searcher, sparse_searcher)
    hybrid_results = evaluate_searcher(
        hybrid_searcher,
        f"Hybrid (α={alpha})",
        k=k,
        alpha=alpha
    )
    results.append(hybrid_results)

# Summary comparison
print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"{'Method':<20} | {'P@10':<6} | {'R@10':<6} | {'MRR':<6} | {'NDCG@10':<6}")
print("-"*60)
for result in results:
    print(f"{result['name']:<20} | {result['precision']:.4f} | {result['recall']:.4f} | {result['mrr']:.4f} | {result['ndcg']:.4f}")