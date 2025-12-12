export JAVA_HOME=/usr/lib/jvm/java-21-openjdk
export PATH=$JAVA_HOME/bin:$PATH

python -m pyserini.encode \
  input \
    --corpus data/bigsample/docs.jsonl\
    --fields text \
    --shard-id 0 \
    --shard-num 1 \
  output \
    --embeddings fullcollectiondenseindex/ \
    --to-faiss \
  encoder \
    --encoder castorini/tct_colbert-v2-hnp-msmarco \
    --fields text \
    --batch 16 \
    --fp16
