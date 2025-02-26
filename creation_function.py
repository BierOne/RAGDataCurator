from typing import Dict, List
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

def make_cosine_dedup_filter(threshold: float):
    """
    Returns a de-duplication function for QA.filter,
    This function uses SentenceTransformer to compute the embedding of a query,
    And embed the current query with a previously "seen" query,
    If the similarity exceeds the threshold, it is considered a duplicate and False is returned.
    Otherwise, save the current query embed and return True.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    seen_embeddings = []  # 用于保存已见查询的嵌入表示

    def cosine_dedup_filter(row: Dict, lang: str = "en") -> bool:
        # 提取查询文本，并去除首尾空白
        query = row.get('query', '').strip()
        if not query:
            # 如果没有有效的 query，则直接过滤掉
            return False

        # 生成当前查询的嵌入表示，转换为二维数组以便计算余弦相似度
        current_embedding = model.encode(query)
        current_embedding = current_embedding.reshape(1, -1)

        # 对比当前查询与已保存的每个嵌入
        for emb in seen_embeddings:
            # emb 已存储为一维数组，这里先调整为二维形式进行计算
            sim = cosine_similarity(current_embedding, emb.reshape(1, -1))[0][0]
            if sim > threshold:
                # 如果与任一已见查询的相似度超过阈值，则认为重复
                return False

        # 如果当前查询与所有已见查询均不相似，则保存该嵌入并保留当前 row
        seen_embeddings.append(current_embedding.flatten())
        return True

    return cosine_dedup_filter



def make_jaccard_dedup_filter(threshold: float):
    """
    Returns a deduplication filter function for use with QA.filter.
    This function computes the Jaccard similarity between the current query and previously seen queries.
    If the similarity exceeds the specified threshold, it considers the current query as a duplicate.
    """
    seen_queries = set()  # Set to store previously encountered queries

    def jaccard_dedup_filter(row: Dict, lang: str = "en") -> bool:
        # Extract the query from the row
        query = row.get('query', '').strip()
        if not query:
            # If the query is empty, consider it a duplicate
            return False

        # Tokenize the query and generate bigrams
        tokens = word_tokenize(query.lower())
        query_bigrams = set(ngrams(tokens, 2))

        # Check for similarity with previously seen queries
        for seen_query in seen_queries:
            seen_tokens = word_tokenize(seen_query.lower())
            seen_bigrams = set(ngrams(seen_tokens, 2))
            intersection = query_bigrams.intersection(seen_bigrams)
            union = query_bigrams.union(seen_bigrams)
            jaccard_similarity = len(intersection) / len(union) if union else 0
            if jaccard_similarity > threshold:
                # If the Jaccard similarity exceeds the threshold, consider it a duplicate
                return False

        # If no duplicates found, add the current query to the seen set
        seen_queries.add(query)
        return True

    return jaccard_dedup_filter


async def add_length_columns(qa_dict: Dict,lang:str='en') -> Dict:
    """An asynchronous handler that adds a length statistical column to QA data"""
    # 处理query长度
    query = qa_dict.get('query', '')
    query_word_count = len(str(query).split())  # 防止非字符串类型
    qa_dict['query_length'] = f"{query_word_count} words"

    # 处理generation_gt长度（支持字符串和列表两种格式）
    generation_gt = qa_dict.get('generation_gt', [])

    # 统一转换为列表处理
    if not isinstance(generation_gt, list):
        generation_gt = [generation_gt]

    gt_lengths = []
    for answer in generation_gt:
        # 处理空值和特殊类型
        clean_answer = str(answer).strip() if answer else ""
        word_count = len(clean_answer.split())
        gt_lengths.append(f"{word_count} words")

    qa_dict['generation_gt_length'] = gt_lengths
    return qa_dict




async def classify_query_type(qa_dict: Dict,lang:str='en') -> Dict:
    """An asynchronous handler that identifies the query type and adds it to the query_type column"""
    query = str(qa_dict.get('query', '')).strip()

    # 定义类型匹配规则（按优先级排序）
    type_patterns = [
        (r'^how\s+many\b', 'how_many'),
        (r'^how\s+much\b', 'how_much'),
        (r'^what\s+type\b', 'what_type'),
        (r'^what\s+kind(s?)', 'what_type'),
        (r'^when\b', 'when'),
        (r'^where\b', 'where'),
        (r'^who\b', 'who'),
        (r'^how\b', 'how_other'),
        (r'^what\b', 'what'),
        (r'^which\b', 'which'),
        (r'^is\s+there\b', 'existence'),
        (r'^does\b', 'yes_no'),
        (r'^can\b', 'capability')
    ]

    # 进行类型匹配
    query_type = 'other'
    for pattern, q_type in type_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            query_type = q_type
            break

    qa_dict['query_type'] = query_type
    return qa_dict