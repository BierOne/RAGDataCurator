from typing import Dict, List
import pandas as pd
import re

async def add_length_columns(qa_dict: Dict,lang:str='en') -> Dict:
    """为QA数据添加长度统计列的异步处理函数"""
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
    """识别查询类型并添加到query_type列的异步处理函数"""
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