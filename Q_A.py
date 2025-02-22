import pandas as pd
from autorag.data.qa.schema import Corpus, Raw, QA
from llama_index.llms.openai import OpenAI
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import make_basic_gen_gt, make_concise_gen_gt
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
from nltk.util import ngrams

from nltk import word_tokenize
import nltk

nltk.download('punkt')


# ================ 自定义全局 Jaccard 过滤函数 ================
def global_jaccard_filter(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """ 全局去重函数（直接操作 DataFrame）"""

    def get_ngram_set(text: str) -> set:
        tokens = word_tokenize(text.lower())
        return set(ngrams(tokens, 2))

    # 提取所有 Query 文本
    queries = df['query'].tolist()
    ngram_sets = [get_ngram_set(q) for q in queries]

    keep_indices = set(range(len(df)))
    for i in range(len(df)):
        if i not in keep_indices:
            continue
        for j in range(i + 1, len(df)):
            if j not in keep_indices:
                continue
            set_i, set_j = ngram_sets[i], ngram_sets[j]
            union = len(set_i | set_j)
            similarity = len(set_i & set_j) / union if union > 0 else 0
            if similarity > threshold:
                keep_indices.discard(j)

    return df.iloc[list(keep_indices)].reset_index(drop=True)


# ================== 主流程 ==================
# 初始化 LLM
query_llm = OpenAI(temperature=0.3)  # 高多样性生成 Query
answer_llm = OpenAI(temperature=0.1)  # 高稳定性生成 Answer

# 加载数据
raw = pd.read_parquet("./parsed_result.parquet")
corpus = pd.read_parquet("./s256_48_100/0.parquet")

# 阶段 1: 生成初始 QA 数据
qa = (
    Corpus(corpus, Raw(raw))
    .sample(random_single_hop,n=100)
    .make_retrieval_gt_contents()
    .batch_apply(factoid_query_gen, llm=query_llm)
)

# 阶段 2: 全局去重（绕过链式调用）
# 将 QA 对象转为 DataFrame 并过滤
qa_df = qa.data  # 直接访问 QA 对象的 DataFrame
filtered_df = global_jaccard_filter(qa_df, threshold=0.8)
qa = QA(filtered_df, qa.linked_corpus)  # 重新包装为 QA 对象

# 阶段 3: 生成 Answer
qa = (
    qa.batch_apply(make_basic_gen_gt, llm=answer_llm)

    .filter(dontknow_filter_rule_based, lang="en")
)

# 保存结果
qa.to_parquet(
    qa_save_path="./data/final_qa.parquet",
    corpus_save_path="./data/final_corpus.parquet"
)

