from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

def make_cosine_dedup_filter(threshold: float):
    """
    返回一个用于 QA.filter 的去重函数，
    该函数利用 SentenceTransformer 计算查询（query）的嵌入，
    并将当前查询与之前“已见”的查询嵌入比较，
    如果相似度超过 threshold，则认为重复，返回 False。
    否则，保存当前查询嵌入并返回 True。
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
