import os
import uuid
import pandas as pd
from typing import Dict, Any
from asyncio import get_event_loop
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
import itertools
from typing import Dict, List

# 假设 process_batch 已经定义或导入
# from autorag.utils.util import process_batch

# 假设 OpenAI 已经正确定义，并支持异步调用
# from your_llm_module import OpenAI

# ----------------------------
# 跨验证函数定义
# ----------------------------
async def cross_validate_qa(row: Dict,llm: BaseLLM) -> Dict:
    """
    对一个 QA 对进行交叉验证：
    根据问题（query）和答案（generation_gt）构造验证 prompt，
    并调用大模型判断答案是否正确且完整。
    验证结果存入 qa_dict 中字段 "validation"。
    """
    # 从 qa_dict 中提取问题和答案
    question = row.get("query", "")
    answer = row.get("generation_gt", "")
    corpus_content =row.get("retrieval_gt_contents", "")
    # 构造验证 prompt
    prompt = (
        "Please verify the following QA pair for factual correctness and completeness based on the provided corpus content.\n\n"
        f"Corpus Content:\n{corpus_content}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Based on the above corpus content, is the answer correct and complete? "
        "Reply with 'YES' if it is, or 'NO' if it is not."
    )
    user_message = ChatMessage(role=MessageRole.USER, content=prompt)
    new_messages = [user_message]
    chat_response: ChatResponse = await llm.achat(messages=new_messages)
    row["validation"] = chat_response.message.content
    return row


# 定义过滤函数，返回 True 表示该 QA 对验证通过
def filter_valid_qa(row: Dict,lang: str = "en") -> bool:
    validation = row.get("validation", "").strip().upper()
    return validation.startswith("YES")





