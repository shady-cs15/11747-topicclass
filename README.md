# 11747-topicclass
topic classification assignment for 11747 spring 2019

# directory structure
download 1m word embeddings from https://fasttext.cc/docs/en/english-vectors.html

{root-dir}/code/\*.py<br/>
{root-dir}/data/topicclass_\*.txt<br/>
{root-dir}/data/wiki-news-300d-1M.vec<br/>  

# running instructions
```
    cd code
    python preprocess.py
    python train_wt_avg.py
    python test_wt_avg.py
```
