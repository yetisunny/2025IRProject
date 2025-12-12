import os
import time
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
from pyserini.search.lucene import LuceneSearcher
import ir_datasets
import numpy as np
import ir_datasets_owi

# Load data
ir_datasets_owi.register()

print("Initializing experiment...")
init_start = time.time()

# Load dataset
dataset = ir_datasets.load("owi/subsampled/dev")

# Build a mapping of query_id -> relevant doc_ids with relevance scores
qrels_dict = {}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

init_time = time.time() - init_start
print(f"Initialization completed in {init_time:.2f} seconds\n")

def evaluate_bm25(k1, b, index_path='../../pyserini_indexes/owi_sample_lucineindex', 
                  k=1000, eval_at_k=[5, 10, 20, 50, 100]):
    """Evaluate BM25 with specific k1 and b parameters"""
    
    # Create searcher with custom BM25 parameters
    searcher = LuceneSearcher(index_path)
    searcher.set_bm25(k1, b)
    
    config_name = f"BM25(k1={k1:.2f}, b={b:.2f})"
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {config_name}")
    print('='*60)
    
    # Timing trackers
    total_search_time = 0
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
        
        # Search with timing
        try:
            search_start = time.time()
            hits = searcher.search(query_text, k=k)
            search_time = time.time() - search_start
            total_search_time += search_time
            
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
    print("TIMING:")
    print(f"{'─'*60}")
    print(f"  Total time:         {overall_time:.2f}s")
    print(f"  Search time:        {total_search_time:.2f}s ({total_search_time/overall_time*100:.1f}%)")
    print(f"  Avg per query:      {overall_time/num_queries:.3f}s")
    print(f"  Queries/second:     {num_queries/overall_time:.2f}")
    
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
        'k1': k1,
        'b': b,
        'name': config_name,
        'mrr': total_mrr/num_queries if num_queries > 0 else 0,
        'total_time': overall_time,
        'search_time': total_search_time,
        'avg_time_per_query': overall_time/num_queries if num_queries > 0 else 0,
        'qps': num_queries/overall_time if overall_time > 0 else 0,
        'num_queries': num_queries,
        **{f'p@{k_val}': metrics_at_k[k_val]['precision']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k},
        **{f'r@{k_val}': metrics_at_k[k_val]['recall']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k},
        **{f'ndcg@{k_val}': metrics_at_k[k_val]['ndcg']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k}
    }

# Run experiments
results = []
experiment_start = time.time()

# Default BM25 parameters (Lucene default: k1=1.2, b=0.75)
print("\n" + "="*80)
print("TESTING DEFAULT AND COMMON BM25 PARAMETER SETTINGS")
print("="*80)

# Test default parameters
results.append(evaluate_bm25(k1=1.2, b=0.75))

# Test variations of k1 (controls term frequency saturation)
# Lower k1 = faster saturation (multiple occurrences matter less)
# Higher k1 = slower saturation (multiple occurrences matter more)
print("\n" + "="*80)
print("VARYING k1 (Term Frequency Saturation)")
print("="*80)
for k1 in [0.6, 0.9, 1.5, 1.8, 2.0]:
    results.append(evaluate_bm25(k1=k1, b=0.75))

# Test variations of b (controls document length normalization)
# b=0: No length normalization
# b=1: Full length normalization
# b=0.75: Standard default (good balance)
print("\n" + "="*80)
print("VARYING b (Document Length Normalization)")
print("="*80)
for b in [0.0, 0.25, 0.5, 1.0]:
    results.append(evaluate_bm25(k1=1.2, b=b))

# Test some combinations
print("\n" + "="*80)
print("TESTING PARAMETER COMBINATIONS")
print("="*80)
combinations = [
    (0.9, 0.4),   # Lower saturation, less length norm
    (1.5, 0.6),   # Higher saturation, moderate length norm
    (0.6, 0.0),   # Low saturation, no length norm
    (2.0, 1.0),   # High saturation, full length norm
]
for k1, b in combinations:
    results.append(evaluate_bm25(k1=k1, b=b))

total_experiment_time = time.time() - experiment_start

# Sort results by NDCG@10 for better comparison
results_sorted = sorted(results, key=lambda x: x['ndcg@10'], reverse=True)

# Summary comparison
print(f"\n{'='*90}")
print("SUMMARY - Performance Metrics (sorted by NDCG@10)")
print('='*90)
print(f"{'Parameters':<20} | {'P@10':<7} | {'R@10':<7} | {'NDCG@10':<8} | {'NDCG@50':<8} | {'MRR':<7}")
print("-"*90)
for result in results_sorted:
    param_str = f"k1={result['k1']:.2f}, b={result['b']:.2f}"
    print(f"{param_str:<20} | {result['p@10']:.4f}  | {result['r@10']:.4f}  | "
          f"{result['ndcg@10']:.4f}   | {result['ndcg@50']:.4f}   | {result['mrr']:.4f}")

print(f"\n{'='*90}")
print("SUMMARY - Timing Analysis")
print('='*90)
print(f"{'Parameters':<20} | {'Total(s)':<9} | {'Search(s)':<10} | {'Avg/Q(s)':<9} | {'Q/s':<7}")
print("-"*90)
for result in results:
    param_str = f"k1={result['k1']:.2f}, b={result['b']:.2f}"
    print(f"{param_str:<20} | {result['total_time']:>8.2f}  | {result['search_time']:>9.2f}  | "
          f"{result['avg_time_per_query']:>8.3f}  | {result['qps']:>6.2f}")

print(f"\n{'='*90}")
print("EXPERIMENT SUMMARY")
print('='*90)
print(f"Total experiment time: {total_experiment_time:.2f}s ({total_experiment_time/60:.1f} minutes)")
print(f"Configurations tested: {len(results)}")
print()

# Find best configuration
best_ndcg10 = max(results, key=lambda x: x['ndcg@10'])
best_mrr = max(results, key=lambda x: x['mrr'])
fastest = max(results, key=lambda x: x['qps'])

print("Best configurations:")
print(f"  Best NDCG@10: k1={best_ndcg10['k1']:.2f}, b={best_ndcg10['b']:.2f} -> {best_ndcg10['ndcg@10']:.4f}")
print(f"  Best MRR:     k1={best_mrr['k1']:.2f}, b={best_mrr['b']:.2f} -> {best_mrr['mrr']:.4f}")
print(f"  Fastest:      k1={fastest['k1']:.2f}, b={fastest['b']:.2f} -> {fastest['qps']:.2f} q/s")
