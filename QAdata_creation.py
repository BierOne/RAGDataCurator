import os
from llama_index.llms.openai import OpenAI
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (make_basic_gen_gt)
from QA_definition import Raw, Corpus,QA,setup_evaluator
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
import pandas as pd
import nltk
from creation_function import (add_length_columns, classify_query_type, make_jaccard_dedup_filter)
from tqdm import tqdm

# load LLM
query_llm = OpenAI(temperature=0.5)
answer_llm = OpenAI(temperature=0.1)
evaluator = setup_evaluator(
    api_key="sk-7d7c8f138a864a08bee5c59e7ca1962b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
# load data
assert os.path.exists("source_data//3//parsed_result.parquet"), "parsed_result.parquet 文件不存在！"
assert os.path.exists("source_data//3//2.parquet"), "0.parquet 文件不存在！"
raw_df = pd.read_parquet("source_data//3//parsed_result.parquet")
raw_instance = Raw(raw_df)
corpus_df = pd.read_parquet("source_data//3//2.parquet")
corpus_instance = Corpus(corpus_df, raw_instance)

empty_qa = QA(
    qa_df=pd.DataFrame(columns=["qid", "query", "query_type","query_length","retrieval_gt","retrieval_gt_contents", "generation_gt","generation_gt_length","ai_eval_score","eval_reason"]),
    linked_corpus=corpus_instance  # 需要传入实际的Corpus实例
)
final_qa = empty_qa

TARGET_SIZE = 3
loop_count = 0
maximum_loops = 50
sample_size=1

while len(final_qa.data)<TARGET_SIZE:
    print(f"\n=== Start round NO. {loop_count+1} of QA data generation ===")
    batch_qa = (
        corpus_instance
        .sample(random_single_hop, n=sample_size)
        .map(lambda df: df.reset_index(drop=True), )
        .make_retrieval_gt_contents()
        .batch_apply(factoid_query_gen, llm=query_llm,)   #query generation
        .batch_apply(make_basic_gen_gt, llm=answer_llm,)  #gt generation
        .filter(dontknow_filter_rule_based, lang="en", )  #filter Garbled gt
        .filter(make_jaccard_dedup_filter(threshold=0.7), lang="en", )  #duplication checking and filter
        .batch_apply(add_length_columns,batch_size=64  )  #query and gt length attribute
        .batch_apply(classify_query_type)                 #query type
        .add_validation(evaluator, score_threshold=75)    #cross validation
    )

    if final_qa.data.empty:
        final_qa = batch_qa
    else:
        final_qa.data = pd.concat([final_qa.data, batch_qa.data],ignore_index=True)
    final_qa.filter(make_jaccard_dedup_filter(threshold=0.7),lang="en",)

    print(f"Current data volume：{len(final_qa.data)}")
    loop_count += 1
    remaining = TARGET_SIZE - len(final_qa.data)
    sample_size = min(40, remaining * 2)  # Dynamically calculate the sample amount

    with tqdm(total=TARGET_SIZE) as pbar:
        pbar.update(len(batch_qa.data))  # Update progress bar

    # Safe exit mechanism
    # if loop_count > maximum_loops:  # Prevent infinite circulation
    #     print("达到最大循环次数")
    #     break

final_qa.to_parquet('output_data//qa.parquet', 'output_data//corpus.parquet')