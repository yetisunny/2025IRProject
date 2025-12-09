import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
import ir_datasets
import numpy as np
import ir_datasets_owi
import gc

# Load data
ir_datasets_owi.register()

# Initialize searchers - ONLY BM25 (no dense retrieval)
sparse_searcher = LuceneSearcher('pyserini_indexes/owi_full_index/')

# Initialize CrossEncoder
print("Loading cross-encoder model...")
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print("Model loaded!")

# Load dataset
dataset = ir_datasets.load("owi/dev")

# Build qrels mapping
qrels_dict = {}
for qrel in dataset.qrels_iter():
    if qrel.query_id not in qrels_dict:
        qrels_dict[qrel.query_id] = {}
    qrels_dict[qrel.query_id][qrel.doc_id] = qrel.relevance

# Memory-efficient document fetching
def get_document_from_index(doc_id, searcher, max_tokens=256):
    """Fetch document directly from Lucene index with truncation"""
    try:
        doc = searcher.doc(doc_id)
        if doc:
            doc_text = doc.raw()
            # Aggressive truncation to save memory
            doc_text = ' '.join(doc_text.split()[:max_tokens])
            return doc_text
        return ""
    except:
        return ""

def evaluate_bm25_with_reranking(k=1000, rerank_top_k=50, eval_at_k=[5, 10, 20, 50, 100]):
    """Evaluate BM25 with cross-encoder reranking - memory optimized"""
    print(f"\n{'='*60}")
    print(f"BM25 + CrossEncoder Reranking (top-{rerank_top_k})")
    print('='*60)
    
    # Track metrics at different k values
    metrics_at_k = {k_val: {'precision': 0.0, 'recall': 0.0, 'ndcg': 0.0} for k_val in eval_at_k}
    total_mrr = 0
    num_queries = 0
    
    queries = list(dataset.queries_iter())
    total_queries = len([q for q in queries if q.query_id in qrels_dict])
    
    print(f"Processing {total_queries} queries...")
    
    for idx, query in enumerate(queries, 1):
        query_id = query.query_id
        query_text = query.text
        
        # Skip if no qrels for this query
        if query_id not in qrels_dict:
            continue
        
        relevant_docs = qrels_dict[query_id]
        relevant_docids = {doc_id for doc_id, rel in relevant_docs.items() if rel > 0}
        
        if len(relevant_docids) == 0:
            continue
        
        # Progress indicator
        if idx % 10 == 0:
            print(f"  Processed {idx}/{len(queries)} queries...")
        
        try:
            # Initial BM25 retrieval
            hits = sparse_searcher.search(query_text, k=k)
            
            # Cross-encoder reranking on top-K
            if len(hits) > 0:
                candidates = hits[:rerank_top_k]
                
                # Prepare query-document pairs
                pairs = []
                valid_hits = []
                for hit in candidates:
                    doc_text = get_document_from_index(hit.docid, sparse_searcher, max_tokens=256)
                    if doc_text:
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
        num_queries += 1
        
        # Periodic garbage collection to free memory
        if idx % 50 == 0:
            gc.collect()
    
    # Print results
    if num_queries > 0:
        print(f"\n{'='*60}")
        print(f"Results over {num_queries} queries:")
        print('='*60)
        print(f"  MRR: {total_mrr/num_queries:.4f}")
        print()
        for k_val in eval_at_k:
            avg_p = metrics_at_k[k_val]['precision'] / num_queries
            avg_r = metrics_at_k[k_val]['recall'] / num_queries
            avg_n = metrics_at_k[k_val]['ndcg'] / num_queries
            print(f"  @{k_val:3d} -> P: {avg_p:.4f}  R: {avg_r:.4f}  NDCG: {avg_n:.4f}")
        
        # Summary table
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"P@5: {metrics_at_k[5]['precision']/num_queries:.4f}")
        print(f"P@10: {metrics_at_k[10]['precision']/num_queries:.4f}")
        print(f"P@20: {metrics_at_k[20]['precision']/num_queries:.4f}")
        print(f"MRR: {total_mrr/num_queries:.4f}")
        print(f"NDCG@10: {metrics_at_k[10]['ndcg']/num_queries:.4f}")
        print(f"R@50: {metrics_at_k[50]['recall']/num_queries:.4f}")
    else:
        print("No queries evaluated!")
    
    return {
        'mrr': total_mrr/num_queries if num_queries > 0 else 0,
        **{f'p@{k_val}': metrics_at_k[k_val]['precision']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k},
        **{f'r@{k_val}': metrics_at_k[k_val]['recall']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k},
        **{f'ndcg@{k_val}': metrics_at_k[k_val]['ndcg']/num_queries if num_queries > 0 else 0 
           for k_val in eval_at_k}
    }

# Run evaluation
print("Starting evaluation...")
print(f"Configuration: BM25 + CrossEncoder (top-50, 256 token truncation)")
results = evaluate_bm25_with_reranking(k=1000, rerank_top_k=50)

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
