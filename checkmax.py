import os
import re
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from autorag.data.qa.schema import QA

# 异步客户端初始化
aclient = AsyncOpenAI(
    api_key="sk-7d7c8f138a864a08bee5c59e7ca1962b",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    timeout=60
)

# 新增评分提示模板
EVAL_PROMPT_TEMPLATE = """根据语料库评估答案质量，给出0-100的AI Eval Score：
[语料库]{corpus}
问题：{question}
答案：{answer}

评分维度：
1. 事实准确性（40%）: 答案是否与语料库信息完全一致
2. 信息完整性（30%）: 是否涵盖语料库中所有相关细节
3. 表述清晰度（20%）: 是否避免歧义表述
4. 相关性（10%）: 是否紧密对应问题

评分规则：
- 90-100: 完全准确且完整，直接引用关键数据
- 80-89: 准确但缺少次要细节
- 70-79: 基本正确但有轻微偏差
- 50-69: 部分正确但遗漏关键信息
- 30-49: 包含错误但有一定相关性
- 0-29: 完全错误或无关

请按格式输出：score|reason
示例：85|答案正确但未提及生产批次信息"""


def parse_eval_response(response):
    """解析评估响应，包含异常处理"""
    try:
        if '|' in response:
            score_part, reason = response.split('|', 1)
        else:
            score_part = response
            reason = "未提供理由"

        # 提取数字
        score = int(re.search(r'\d+', score_part).group())
        return max(0, min(100, score)), reason.strip()
    except Exception as e:
        print(f"解析评分失败: {e} | 原始响应: {response}")
        return 50, "评分解析失败"


# 重试装饰器配置（指数退避）
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_eval_answer(client, prompt):
    response = await client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50,
        temperature=0
    )
    return response.choices[0].message.content.strip()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def async_check_source(client, prompt):
    response = await client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0
    )
    return response.choices[0].message.content.strip()


async def process_single_qa(row, corpus_text, semaphore):
    async with semaphore:
        question = row['query']
        answer = row.get('generation_gt', '')

        # 构建评估提示词
        eval_prompt = EVAL_PROMPT_TEMPLATE.format(
            corpus=corpus_text[:10000],  # 控制上下文长度
            question=question,
            answer=answer
        )

        source_prompt = f"""找出答案依据：
        [语料库]{corpus_text[:10000]}
        问题：{question}
        答案：{answer}
        要求：直接引用支持答案的原文"""

        try:
            # 并行执行评估和溯源
            eval_task = async_eval_answer(aclient, eval_prompt)
            source_task = async_check_source(aclient, source_prompt)
            eval_res, source = await asyncio.gather(eval_task, source_task)

            # 解析评分
            score, reason = parse_eval_response(eval_res)

            # 溯源结果处理
            source = source if len(source) > 10 else '未找到'
        except Exception as e:
            print(f"处理出错：{e}")
            score, reason, source = 50, "评估失败", "error"

        return pd.Series([score, reason, source],
                         index=['ai_eval_score', 'eval_reason', 'answer_source'])


async def batch_process(qa_df, corpus_text, batch_size=50, max_concurrency=10):
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    for i in range(0, len(qa_df), batch_size):
        batch = qa_df.iloc[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(qa_df) - 1) // batch_size + 1}")

        tasks = [process_single_qa(row, corpus_text, semaphore) for _, row in batch.iterrows()]
        batch_results = await asyncio.gather(*tasks)

        # 实时保存带评估结果
        batch_df = pd.concat([batch.reset_index(drop=True),
                              pd.concat(batch_results, axis=1).T], axis=1)
        save_path = f"./old_data/batch_{i // batch_size}_eval.parquet"
        batch_df.to_parquet(save_path)
        print(f"Batch saved with eval scores: {save_path}")

        results.extend(batch_results)

    return pd.concat(results, axis=1).T


def main():
    # 加载数据
    qa_df = pd.read_parquet("./s256_48_100/pc3/final_qa.parquet")
    corpus = pd.read_parquet("./s256_48_100/pc3/final_corpus.parquet")
    corpus_text = "\n".join(corpus['contents'].tolist())

    # 执行异步处理
    loop = asyncio.get_event_loop()
    result_df = loop.run_until_complete(batch_process(qa_df, corpus_text))

    # 合并结果
    final_df = qa_df.join(result_df)

    # 添加质量分级
    final_df['quality_tier'] = pd.cut(final_df['ai_eval_score'],
                                      bins=[-1, 29, 49, 69, 89, 100],
                                      labels=['E', 'D', 'C', 'B', 'A'])

    final_df.to_parquet("./s256_48_100/pc3/final_qa_evaluated.parquet")
    print(f"处理完成，新增评估维度：{final_df.columns.tolist()}")


if __name__ == "__main__":
    main()
