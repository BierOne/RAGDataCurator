from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatResponse, ChatMessage, MessageRole
from typing import Dict, List


async def cross_validate_qa(row: Dict, row2:dict, llm: BaseLLM, threshold: float = 80) -> Dict:
    """
    对一个 QA 对进行交叉验证：
    根据问题（query）和答案（generation_gt）构造验证 prompt，
    并调用大模型对答案进行打分。得分越接近 1，表示答案越接近正确答案。
    答案的分数保存在 "score" 字段中。
    如果得分低于阈值，则标记为需要删除的答案。
    """
    # 从 qa_dict 中提取问题、答案和语料库内容
    question = row.get("query", "")
    answer = row.get("generation_gt", "")
    corpus_content = row2.get("content", "")


    # 构造验证 prompt
    prompt = (
        "Please verify the following QA pair for factual correctness and completeness based on the provided corpus content.\n\n"
        f"Corpus Content:\n{corpus_content}\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Based on the above corpus content, rate the answer on a float number scale from 0 to 100, where 100 is a perfect match to the correct answer."
        "Reply with the score."
    )

    # 发送给大模型并获取响应
    user_message = ChatMessage(role=MessageRole.USER, content=prompt)
    new_messages = [user_message]
    chat_response: ChatResponse = await llm.achat(messages=new_messages)

    # 获取返回的分数并保存在 "score" 字段中
    try:
        score = float(chat_response.message.content.strip())  # 尝试将返回的内容转换为 float
    except ValueError:
        score = 0.0  # 如果转换失败，设置得分为 0

    # 将分数存入 row
    row["score"] = score


    # 如果分数低于阈值，认为该条 QA 需要被删除（或者进行进一步处理）
    if score < threshold:
        row["validation"] = "DELETE"  # 可以根据需要设置为 'DELETE' 或其他标记
    else:
        row["validation"] = "PASS"

    return row

def filter_valid_qa(row: Dict, lang: str = "en", threshold: float = 80) -> bool:
    """
    过滤掉得分低于阈值的 QA 对。
    如果得分低于阈值，返回 False，表示该 QA 对不符合要求。
    """
    score = row.get("score", 0.0)  # 获取得分
    return score >= threshold  # 得分大于等于阈值的 QA 对保留