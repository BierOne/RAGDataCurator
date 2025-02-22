import os
import pandas as pd
from llama_index.llms.openai import OpenAI
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (make_basic_gen_gt,make_concise_gen_gt,)
from autorag.data.qa.schema import Raw, Corpus,QA
from autorag.data.qa.query.llama_gen_query import factoid_query_gen, two_hop_incremental
from autorag.data.qa.sample import (random_single_hop,range_single_hop)
from two_hop_sample import random_two_hop
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncClient, base_url
from sklearn.metrics.pairwise import cosine_similarity
from cross_val import (cross_validate_qa,filter_valid_qa)

llm = OpenAI(tempreture=0.5)
# 加载数据
assert os.path.exists("2//parsed_result.parquet"), "parsed_result.parquet 文件不存在！"
assert os.path.exists("2//0.parquet"), "0.parquet 文件不存在！"
raw_df = pd.read_parquet("2//parsed_result.parquet")
raw_df=raw_df.loc[0:37]
raw_instance = Raw(raw_df)
corpus_df = pd.read_parquet("2//0.parquet")
corpus_df=corpus_df.loc[0:47]
corpus_instance = Corpus(corpus_df, raw_instance)



initial_qa = (
    corpus_instance
    #.sample(random_single_hop,n=30)
    .sample(random_two_hop,n=40)
    .map(lambda df: df.reset_index(drop=True),)
    .make_retrieval_gt_contents()
    # .batch_apply(
    #     factoid_query_gen,  # query generation
    #     llm=llm,
    # )
    .batch_apply(
        two_hop_incremental,
        llm=llm
    )
    .batch_apply(
        make_basic_gen_gt,  # answer generation (basic)
        llm=llm,
    )
    .filter(dontknow_filter_rule_based,lang="en",)
    .batch_apply(cross_validate_qa,llm=llm)
    .filter(filter_valid_qa,lang="en",)
)

if not isinstance(initial_qa.data, pd.DataFrame):
    raise ValueError("initial qa.data must be a pandas DataFrame.")
#假设你已经生成了、initial_ga~，并从中提取了查询部分
queries = initial_qa.data['query'].tolist() # 替换为你的实际方法提取 query 部分
#获取查询的嵌入表示
# 基于嵌入相似度的同步版本
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(queries)
#计算查询之间的余弦相似度
cosine_sim= cosine_similarity(embeddings)
#输出余弦相似度矩阵
print("cosine similarity Matrix:\n",cosine_sim)
#根据设定的阈值过滤掉相似的查询
threshold =0.6 #设定相似度阈值，超过此值的查询会被认为是相似的
filtered_indices =[]

for i,query in enumerate(queries):
    is_similar = False
    for j in range(i):
        if cosine_sim[i][j]>threshold:
            is_similar = True
        break
    if not is_similar:
        filtered_indices.append(i)
#根据过滤的索引提取 DataFrame 的子集
filtered_data = initial_qa.data.iloc[filtered_indices].copy()#使用 .copy()是为了创建一个新的独立副本，而不是直接对原始 initia1 ga.data 的视图进行操作。这样可以避免在修改 filtered data 时影响到原始数据。
print("\nFiltered Queries:")
print(type(filtered_data))
#将更新后的数据替代 initial qa.data
initial_qa.data = filtered_data

initial_qa.to_parquet('two_hop_qadata.parquet', 'two_hop_corpus.parquet')