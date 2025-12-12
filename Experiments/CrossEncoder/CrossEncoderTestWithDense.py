import os
import time
from collections import defaultdict

from pyserini.search.hybrid import HybridSearcher
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
print("Initializing searchers...")
init_start = time.time()
dense_searcher = FaissSearcher('colbert_encoded_docs/', 'castorini/tct_colbert-v2-hnp-msmarco')
sparse_searcher = LuceneSearcher('pyserini_indexes/owi_sample_lucineindex')
hybrid_searcher = HybridSearcher(dense_searcher, sparse_searcher)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
init_time = time.time() - init_start
print(f"Initialization completed in {init_time:.2f} seconds\n")

# Load dataset
dataset = ir_datasets.load("owi/subsampled/dev")

# Build a mapping of query_id -> relevant doc_ids with relevance scores
qrels_dict = {}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

def get_document_from_index(doc_id, searcher):
    """Fetch document directly from Lucene index"""
    try:
        doc = searcher.doc(doc_id)
        if doc:
            return doc.raw()
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
    
    # Timing trackers
    total_search_time = 0
    total_rerank_time = 0
    total_doc_fetch_time = 0
    total_eval_time = 0
    
    # Track metrics at different k values
    metrics_at_k = {k_val: {'precision': 0, 'recall': 0, 'ndcg': 0} for k_val in eval_at_k}
    total_mrr = 0
    num_queries = 0
    
    overall_start = time.time()
    
    for query in dataset.queries_iter():
        query_id = query.query_id
        query_text = query.text
        
        # Skip if no qrels for this query
        if query_id not in qrels_dict:
            continue
        
        relevant_docs = qrels_dict[query_id]
        relevant_docids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
        
        if len(relevant_docids) == 0:
            continue
        
        # Initial retrieval with timing
        try:
            search_start = time.time()
            hits = searcher.search(query_text, k=k)
            search_time = time.time() - search_start
            total_search_time += search_time
            
            # Apply cross-encoder reranking if enabled
            if use_reranker and len(hits) > 0:
                rerank_start = time.time()
                
                # Take top rerank_top_k candidates for reranking
                candidates = hits[:rerank_top_k]
                
                # Prepare query-document pairs for cross-encoder
                pairs = []
                valid_hits = []
                
                doc_fetch_start = time.time()
                for hit in candidates:
                    doc_text = get_document_from_index(hit.docid, sparse_searcher)
                    if doc_text:
                        doc_text = ' '.join(doc_text.split()[:512])
                        pairs.append([query_text, doc_text])
                        valid_hits.append(hit)
                doc_fetch_time = time.time() - doc_fetch_start
                total_doc_fetch_time += doc_fetch_time
                
                if len(pairs) > 0:
                    # Get cross-encoder scores
                    ce_start = time.time()
                    ce_scores = reranker.predict(pairs)
                    ce_time = time.time() - ce_start
                    
                    # Sort by cross-encoder scores
                    scored_hits = list(zip(valid_hits, ce_scores))
                    scored_hits.sort(key=lambda x: x[1], reverse=True)
                    
                    # Reorder hits
                    reranked_hits = [hit for hit, score in scored_hits]
                    remaining_hits = hits[rerank_top_k:]
                    hits = reranked_hits + remaining_hits
                
                rerank_time = time.time() - rerank_start
                total_rerank_time += rerank_time
            
            retrieved_docids = [hit.docid for hit in hits]
        except Exception as e:
            print(f"Error searching for query {query_id}: {e}")
            continue
        
        # Calculate metrics at different k values
        eval_start = time.time()
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
            
            # Ideal ranking
            sorted_rels = sorted(relevant_docs.values(), reverse=True)[:k_val]
            idcg = 0
            for i, rel in enumerate(sorted_rels, 1):
                idcg += rel / np.log2(i + 1)
            
            ndcg_at_k = dcg / idcg if idcg > 0 else 0
            
            metrics_at_k[k_val]['precision'] += precision_at_k
            metrics_at_k[k_val]['recall'] += recall_at_k
            metrics_at_k[k_val]['ndcg'] += ndcg_at_k
        
        # MRR
        rr = 0
        for i, docid in enumerate(retrieved_docids, 1):
            if docid in relevant_docids:
                rr = 1 / i
                break
        
        total_mrr += rr
        eval_time = time.time() - eval_start
        total_eval_time += eval_time
        num_queries += 1
    
    overall_time = time.time() - overall_start
    
    # Print timing breakdown
    print(f"\n{'─'*60}")
    print("TIMING BREAKDOWN:")
    print(f"{'─'*60}")
    print(f"  Total time:           {overall_time:.2f}s")
    print(f"  Search time:          {total_search_time:.2f}s ({total_search_time/overall_time*100:.1f}%)")
    if use_reranker:
        print(f"  Reranking time:       {total_rerank_time:.2f}s ({total_rerank_time/overall_time*100:.1f}%)")
        print(f"    - Doc fetching:     {total_doc_fetch_time:.2f}s")
        print(f"    - CrossEncoder:     {total_rerank_time-total_doc_fetch_time:.2f}s")
    print(f"  Evaluation time:      {total_eval_time:.2f}s ({total_eval_time/overall_time*100:.1f}%)")
    print(f"  Avg time per query:   {overall_time/num_queries:.3f}s")
    if use_reranker:
        print(f"  Avg rerank per query: {total_rerank_time/num_queries:.3f}s")
    
    # Print average metrics
    if num_queries > 0:
        print(f"\n{'─'*60}")
        print(f"RESULTS over {num_queries} queries:")
        print(f"{'─'*60}")
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
        'total_time': overall_time,
        'search_time': total_search_time,
        'rerank_time': total_rerank_time,
        'doc_fetch_time': total_doc_fetch_time,
        'eval_time': total_eval_time,
        'avg_time_per_query': overall_time/num_queries if num_queries > 0 else 0,
        'num_queries': num_queries,
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

experiment_start = time.time()

# BM25 only
bm25_results = evaluate_searcher(sparse_searcher, "BM25 (Lucene)", k=k, use_reranker=False)
results.append(bm25_results)

# Dense only
dense_results = evaluate_searcher(dense_searcher, "Dense (FAISS)", k=k, use_reranker=False)
results.append(dense_results)

# Hybrid only
hybrid_results = evaluate_searcher(hybrid_searcher, "Hybrid (bm25 + colbert dense fusion)", k=k, use_reranker=False)
results.append(hybrid_results)

# Hybrid + Cross-Encoder Reranking
hybrid_rerank_results = evaluate_searcher(hybrid_searcher, "Hybrid + CrossEncoder", k=k, use_reranker=True, rerank_top_k=100)
results.append(hybrid_rerank_results)

# BM25 + Cross-Encoder Reranking
bm25_rerank_results = evaluate_searcher(sparse_searcher, "BM25 + CrossEncoder", k=k, use_reranker=True, rerank_top_k=100)
results.append(bm25_rerank_results)

# Dense + Cross-Encoder Reranking
dense_rerank_results = evaluate_searcher(dense_searcher, "Dense + CrossEncoder", k=k, use_reranker=True, rerank_top_k=100)
results.append(dense_rerank_results)

# Optional: Try different rerank_top_k values for BM25
for rerank_k in [50, 200]:
    bm25_rerank = evaluate_searcher(
        sparse_searcher,
        f"BM25 + CE (top-{rerank_k})",
        k=k,
        use_reranker=True,
        rerank_top_k=rerank_k
    )
    results.append(bm25_rerank)

total_experiment_time = time.time() - experiment_start

# Summary comparison
print(f"\n{'='*80}")
print("SUMMARY - Performance Metrics")
print('='*80)
print(f"{'Method':<25} | {'P@5':<7} | {'P@10':<7} | {'NDCG@10':<7} | {'MRR':<7}")
print("-"*80)
for result in results:
    print(f"{result['name']:<25} | {result['p@5']:.4f}  | {result['p@10']:.4f}  | "
          f"{result['ndcg@10']:.4f}  | {result['mrr']:.4f}")

print(f"\n{'='*80}")
print("SUMMARY - Timing Analysis")
print('='*80)
print(f"{'Method':<25} | {'Total(s)':<9} | {'Search(s)':<10} | {'Rerank(s)':<10} | {'Avg/Q(s)':<9}")
print("-"*80)
for result in results:
    rerank_str = f"{result['rerank_time']:.2f}" if result['rerank_time'] > 0 else "-"
    print(f"{result['name']:<25} | {result['total_time']:>8.2f}  | {result['search_time']:>9.2f}  | "
          f"{rerank_str:>9}  | {result['avg_time_per_query']:>8.3f}")

print(f"\n{'='*80}")
print("TIMING SUMMARY")
print('='*80)
print(f"Total experiment time: {total_experiment_time:.2f}s ({total_experiment_time/60:.1f} minutes)")
print(f"Initialization time:   {init_time:.2f}s")
print(f"\nSpeed comparison (queries/second):")
for result in results:
    qps = result['num_queries'] / result['total_time']
    print(f"  {result['name']:<30}: {qps:>6.2f} q/s")
