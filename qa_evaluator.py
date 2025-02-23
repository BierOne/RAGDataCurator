import re
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import Dict, List
from autorag.data.qa.schema import QA,Corpus





class QAEvaluator:
    def __init__(self, api_key, base_url, model="qwen-plus", max_concurrency=8):
        """
        初始化评估器
        :param api_key: OpenAI API密钥
        :param base_url: API基础URL
        :param model: 使用的模型名称，默认qwen-plus
        :param max_concurrency: 最大并发数，默认8
        """
        self._configure_logging()

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=60
        )
        self.model = model
        self.max_concurrency = max_concurrency
        self.eval_prompt_template = """请执行三阶段评估：

[阶段1：依据核查]
语料库片段：{corpus}
问题：{question}
参考答案：{answer}

核查要求：
1. 检查答案是否在语料中有直接支持（直接匹配）
2. 若无直接支持，检查是否可通过语料合理推断（逻辑匹配）
3. 记录匹配类型（直接/推断/无）
[阶段2：逻辑验证]
若为推断类型，必须验证：
1. 推理前提是否全部来自语料
2. 推理步骤是否符合逻辑规则
3. 结论是否必然成立
[阶段3:评分]
评分维度：
- 核心准确性（60%）：关键事实是否正确
- 表述合理性（30%）：是否清晰无歧义
- 信息完整性（10%）：是否涵盖必要细节

评分规则：
- 直接匹配：基准分90±10
- 正确推断：基准分75±15
- 推断中仅部分正确：基准分60±10
- 错误推断：基准分30±20
输出格式：分数|匹配类型|评分理由(至少50字）

示例：
92|直接|答案完全准确引用语料第三段内容："...". 包含所有技术参数和有效期说明。
90|推断|正确推导国家数量，答案完全正确.
43|无|声称支持银联支付，但语库中未见相关支付方式记载
60|推断|答案中说有12种蔬菜，但其中八种是海鲜鱼类，只有四种是蔬菜，答案部分正确"""  # 保持原模板内容不变

    def _configure_logging(self):
        """配置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("evaluation.log"),
                logging.StreamHandler()
            ]
        )

    @staticmethod
    def dynamic_context_selection(question, answer, full_corpus, window_size=3000):

        # 提取特征

        keywords = set(re.findall(r'\w{2,}', question.lower()))
        entities = set(re.findall(r'[A-Z0-9\-_]{3,}', answer))  # 匹配型号等标识

        best_score = 0
        best_start = 0

    # 滑动窗口扫描
        for i in range(0, len(full_corpus), 500):
            segment = full_corpus[i:i + window_size]
            seg_keywords = set(re.findall(r'\w{2,}', segment.lower()))
            seg_entities = set(re.findall(r'[A-Z0-9\-_]{3,}', segment))

        # 计算匹配度
            kw_match = len(keywords & seg_keywords)
            entity_match = len(entities & seg_entities)
            score = kw_match * 1 + entity_match * 3

            if score > best_score or (score == best_score and i < best_start):
                best_score = score
                best_start = i

        selected = full_corpus[best_start:best_start + window_size]
        return selected if len(selected) > 500 else full_corpus[:window_size]


    @staticmethod
    def parse_eval_response(response):
        
        # ... 原函数实现 ...
        try:  # 如果响应为空或明显无效，直接返回默认值
            if not response or "Invalid" in response or "无法" in response:
                raise ValueError("无效响应")
            # 清洗响应内容
            clean_res = re.sub(r'\s+', ' ', response).strip()[:500]

            # 提取分数部分
            score_match = re.search(r'(\d{1,3})[\s分]*\|', clean_res)
            if not score_match:
                score_match = re.search(r'^(\d{1,3})', clean_res)

            score = int(score_match.group(1)) if score_match else 50

            # 提取类型和理由
            parts = re.split(r'[\|]', clean_res)
            match_type = '无'
            reason = '未说明'

            if len(parts) >= 3:
                match_type = '直接' if '直接' in parts[1] else '推断' if '推断' in parts[1] else '无'
                reason = parts[2].strip()  # 除前后空格
            elif len(parts) == 2:
                if any(w in parts[1] for w in ['直接', '推断']):
                    match_type = '直接' if '直接' in parts[1] else '推断'
                    reason = '未说明'
                else:
                    reason = parts[1].strip()[:20]  # 去除前后空格

            # 分数校准
            score = max(0, min(100, score))
            if match_type == '无' and score > 40:
                score = min(40, score)

            return score, f"{match_type}|{reason}"  # 确保|后无空格

        except Exception as e:
            logging.error(f"解析失败: {e} | 原始响应: {response}")
            return 50, "解析异常"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=20))
    async def _async_eval_answer(self, prompt):
        """强化版评估函数，增加输出校验"""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
            raw_output = response.choices[0].message.content.strip()

            if not raw_output or "抱歉" in raw_output or "无法" in raw_output:
                raise ValueError("无效响应")
            return raw_output
        except Exception as e:
            logging.warning(f"评估请求失败: {e}")
            raise

    async def _process_single_qa(self, row, corpus_text, semaphore):
        """处理单个QA对"""
        async with semaphore:
            try:
                question = str(row['query'])
                answer = str(row.get('generation_gt', ''))

                context = self.dynamic_context_selection(question, answer, corpus_text)
                eval_prompt = self.eval_prompt_template.format(
                    corpus=context[:5000],
                    question=question,
                    answer=answer
                )

                raw_response = await self._async_eval_answer(eval_prompt)
                score, eval_reason = self.parse_eval_response(raw_response)

                return pd.Series([score, eval_reason],
                                 index=['ai_eval_score', 'eval_reason'])

            except Exception as e:
                logging.error(f"处理异常：{e}")
                return pd.Series([50, '处理失败'],
                                 index=['ai_eval_score', 'eval_reason'])

    # async def evaluate(self, qa_df, corpus_text, batch_size=20):
    #     """
    #     执行批量评估
    #     :param qa_df: 包含query和generation_gt的DataFrame
    #     :param corpus_text: 语料库全文
    #     :param batch_size: 批处理大小，默认20
    #     :return: 包含评估结果的DataFrame
    #     """
    #     semaphore = asyncio.Semaphore(self.max_concurrency)
    #     results = []
    #
    #     for i in range(0, len(qa_df), batch_size):
    #         batch = qa_df.iloc[i:i + batch_size]
    #         logging.info(f"Processing batch {i // batch_size + 1}/{(len(qa_df) - 1) // batch_size + 1}")
    #
    #         tasks = [self._process_single_qa(row, corpus_text, semaphore)
    #                  for _, row in batch.iterrows()]
    #         batch_results = await asyncio.gather(*tasks)
    #         results.extend(batch_results)
    #
    #     return pd.concat(results, axis=1).T
    async def evaluate(self, qa_df, corpus_text, batch_size=20, score_threshold=None):
        """
        执行批量评估（新增score_threshold参数）
        :param score_threshold: 分数阈值，默认不过滤
        """
        semaphore = asyncio.Semaphore(self.max_concurrency)
        results = []

        for i in range(0, len(qa_df), batch_size):
            batch = qa_df.iloc[i:i + batch_size]
            logging.info(f"Processing batch {i // batch_size + 1}/{(len(qa_df) - 1) // batch_size + 1}")

            tasks = [self._process_single_qa(row, corpus_text, semaphore)
                     for _, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        # 合并结果并添加过滤逻辑
        result_df = pd.concat(results, axis=1).T
        final_df = qa_df.join(result_df)

        if score_threshold is not None:
            final_df = final_df[final_df['ai_eval_score'] >= score_threshold]

        return final_df
    @staticmethod
    def generate_report(df):
        """生成质量分析报告"""
        report = {
            'total': len(df),
            'avg_score': df['ai_eval_score'].mean(),
            'score_distribution': df['ai_eval_score'].value_counts().sort_index(),
            'common_reasons': df['eval_reason'].value_counts().head(5)
        }

        print(f"\n{'=' * 40}\n质量分析报告\n{'=' * 40}")
        print(f"平均分：{report['avg_score']:.1f}")
        print("\n分数分布：")
        print(report['score_distribution'].to_string())
        print("\n常见评估原因：")
        print(report['common_reasons'].to_string())

    def save_report(self, df, output_path, score_threshold=None):
        """
        新增保存方法（支持阈值过滤）
        :param score_threshold: 保存时应用的分数阈值
        """
        if score_threshold is not None:
            df = df[df['ai_eval_score'] >= score_threshold]

        df.to_parquet(output_path)
        logging.info(f"已保存结果到：{output_path}（条目数：{len(df)}）")
    # @staticmethod
    # def test_parse_logic():
    #     """（可选）解析逻辑测试"""
    #     # ... 保持原测试内容不变 ...
    #     """解析逻辑测试"""
    #     test_cases = [
    #         ("85|直接|数据准确", (85, "直接|数据准确")),
    #         ("90 | 推断 | 合理推论", (90, "推断|合理推论")),  # 去除空格
    #         ("75分|直接", (75, "直接|未说明")),
    #         ("Invalid response", (50, "解析异常")),  # 无效响应
    #         ("无法评估", (50, "解析异常")),  # 无效响应
    #         ("60|无|部分匹配", (40, "无|部分匹配"))  # 测试分数校准
    #     ]
    #
    #     for input_, expected in test_cases:
    #         output = parse_eval_response(input_)
    #         assert output == expected, f"测试失败：{input_} => {output}"


def setup_evaluator(api_key: str, base_url: str) -> QAEvaluator:
    """初始化评估器工厂函数"""
    return QAEvaluator(
        api_key=api_key,
        base_url=base_url,
        model="qwen-plus",
        max_concurrency=8
    )




# 示例用法
async def cross_validate_qa(
        qa_dict: Dict,
        evaluator: QAEvaluator,
        linked_corpus: Corpus
) -> Dict:
    """集成化单条QA验证函数"""
    try:
        # 获取关联语料内容
        corpus_text = "\n".join(
            str(c) for c in linked_corpus.data['contents'].tolist()
            if pd.notnull(c)
        )

        # 创建临时DataFrame
        temp_df = pd.DataFrame([qa_dict])

        # 执行评估
        evaluated_df = await evaluator.evaluate(
            temp_df,
            corpus_text,
            batch_size=1,  # 单条处理
            score_threshold=None  # 保留所有结果
        )

        # 合并结果
        return {
            **qa_dict,
            'ai_eval_score': evaluated_df.iloc[0]['ai_eval_score'],
            'eval_reason': evaluated_df.iloc[0]['eval_reason']
        }
    except Exception as e:
        print(f"验证失败 qid={qa_dict.get('qid')}: {str(e)}")
        return {
            **qa_dict,
            'ai_eval_score': 0,
            'eval_reason': '验证异常'
        }