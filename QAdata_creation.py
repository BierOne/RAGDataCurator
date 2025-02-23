import os
from llama_index.llms.openai import OpenAI
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (make_basic_gen_gt,make_concise_gen_gt,)
from autorag.data.qa.schema import Raw, Corpus,setup_evaluator
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import AsyncClient
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from cosine_sim_filter import (make_cosine_dedup_filter,make_jaccard_dedup_filter)
import nltk
from get_column_attribute import (add_length_columns,classify_query_type)


def count_words(text: str) -> int:
    """计算给定文本的单词数。"""
    return len(text.split())


query_llm = OpenAI(temperature=0.5)
answer_llm = OpenAI(temperature=0.1)


# 加载数据
assert os.path.exists("data//3//parsed_result.parquet"), "parsed_result.parquet 文件不存在！"
assert os.path.exists("data//3//2.parquet"), "0.parquet 文件不存在！"
raw_df = pd.read_parquet("data//3//parsed_result.parquet")
#raw_df=raw_df.loc[0:37]
raw_instance = Raw(raw_df)
corpus_df = pd.read_parquet("data//3//2.parquet")
#corpus_df=corpus_df.loc[0:47]
corpus_instance = Corpus(corpus_df, raw_instance)

evaluator = setup_evaluator(
    api_key="sk-7d7c8f138a864a08bee5c59e7ca1962b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

initial_qa = (
    corpus_instance
    .sample(random_single_hop,n=100)
    .map(lambda df: df.reset_index(drop=True),)
    .make_retrieval_gt_contents()
    .batch_apply(
        factoid_query_gen,  # query generation
        llm=query_llm,
    )
    .batch_apply(
        make_basic_gen_gt,  # answer generation (basic)
        llm=answer_llm,
    )
    .filter(dontknow_filter_rule_based,lang="en",)
    .filter(make_jaccard_dedup_filter(threshold=0.7),lang="en",)
    .batch_apply(
        add_length_columns,
        batch_size=64  # 根据数据量调整批次大小
    )
    .batch_apply(classify_query_type)
    .add_validation(evaluator, score_threshold=75)
)

print(initial_qa.data['query_type'].value_counts())
final_qa = initial_qa


while len(final_qa.data)<500:
    print("进入循环")
    after_qa = (
        corpus_instance
        .sample(random_single_hop, n=100)
        .map(lambda df: df.reset_index(drop=True), )
        .make_retrieval_gt_contents()
        .batch_apply(
            factoid_query_gen,  # query generation
            llm=query_llm,
        )
        .batch_apply(
            make_basic_gen_gt,  # answer generation (basic)
            llm=answer_llm,
        )
        .filter(dontknow_filter_rule_based, lang="en", )
        .filter(make_jaccard_dedup_filter(threshold=0.7), lang="en", )
        .batch_apply(
            add_length_columns,
            batch_size=64  # 根据数据量调整批次大小
        )
        .batch_apply(classify_query_type)
        .add_validation(evaluator, score_threshold=75)
    )
    final_qa.data = pd.concat([final_qa.data, after_qa.data], ignore_index=True)
    final_qa.filter(make_jaccard_dedup_filter(threshold=0.7),lang="en",)
    #将更新后的数据替代 after qa.data
    print("结束一次循环,qa data目前数量为")
    print(len(final_qa.data))

final_qa.to_parquet('output_data//qa.parquet', 'output_data//corpus.parquet')