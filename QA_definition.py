import logging
from typing import Callable, Optional, Dict, Awaitable, Any, Tuple, List
import uuid
import pandas as pd
from autorag.utils.util import process_batch, get_event_loop, fetch_contents
from autorag.support import get_support_modules
import re
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from typing import Dict, List


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
        self.eval_prompt_template = """请执行两阶段评估：

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
        """保持原dynamic_context_selection函数内容不变"""
        # ... 原函数实现 ...
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
        """保持原parse_eval_response函数内容不变"""
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

logger = logging.getLogger("AutoRAG")


class Raw:
	"""
	The Raw class that stored document parsing results.
	It can do chunking.
	It has two column names, 'raw_id' and 'contents'.
	"""

	def __init__(self, raw_df: Optional[pd.DataFrame] = None):
		self.data = raw_df

	def batch_apply(
		self, fn: Callable[[Dict, Any], Awaitable[Dict]], batch_size: int = 32, **kwargs
	) -> "Raw":
		raw_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(raw_dict, **kwargs) for raw_dict in raw_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return Raw(pd.DataFrame(results))

	def map(self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs) -> "Raw":
		return Raw(fn(self.data, **kwargs))

	def flatmap(self, fn: Callable, **kwargs) -> "Raw":
		return fn(self.data, **kwargs)

	def chunk(self, module_name: str, **module_params) -> "Corpus":
		chunk_module = get_support_modules(module_name)
		chunked_result = chunk_module(parsed_result=self.data, **module_params)
		return Corpus(chunked_result, self)

	def __add__(self, other):
		assert isinstance(other, Raw), "You can only add Raw instances."
		self.data = pd.concat([self.data, other.data], ignore_index=True).reset_index(
			drop=True
		)
		return self


class Corpus:
	"""
	The Corpus class that stored chunked passages.
	It can generate qa set, linked with Raw instance.
	"""

	def __init__(
		self,
		corpus_df: Optional[pd.DataFrame] = None,
		linked_raw: Optional[Raw] = None,
	):
		self.data = corpus_df
		self._linked_raw = linked_raw

	@property
	def linked_raw(self) -> Raw:
		return self._linked_raw

	@linked_raw.setter
	def linked_raw(self, raw: Raw):
		raise NotImplementedError("linked_raw is read-only.")

	def to_parquet(self, save_path: str):
		"""
		Save the corpus to the AutoRAG compatible parquet file.
		It is not for the data creation, for running AutoRAG.
		If you want to save it directly, use the below code.
		`corpus.data.to_parquet(save_path)`

		:param save_path: The path to save the corpus.
		"""
		if not save_path.endswith(".parquet"):
			raise ValueError("save_path must be ended with .parquet")
		save_df = self.data.reset_index(drop=True)
		save_df.to_parquet(save_path)

	def batch_apply(
		self, fn: Callable[[Dict, Any], Awaitable[Dict]], batch_size: int = 32, **kwargs
	) -> "Corpus":
		corpus_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(corpus_dict, **kwargs) for corpus_dict in corpus_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))
		return Corpus(pd.DataFrame(results), self.linked_raw)

	def map(
		self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs
	) -> "Corpus":
		return Corpus(fn(self.data, **kwargs), self.linked_raw)

	def sample(self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs) -> "QA":
		"""
		Sample the corpus for making QA.
		It selects the subset of the corpus and makes QA set from it.
		You can generate questions from the created question.
		It is the first step to make QA set from the corpus.
		If you select just one passage from each passage, it will be a single-hop QA set.
		If you select multiple passages from each passage, it will be a multi-hop QA set.

		:param fn: The select function to perform.
		It returns QA dataframe.
		:return: QA instance that is selected.
		It contains qid and retrieval_gt columns.
		"""
		return QA(fn(self.data, **kwargs), self)


class QA:
	def __init__(
		self,
		qa_df: Optional[pd.DataFrame] = None,
		linked_corpus: Optional[Corpus] = None,
	):
		self.data = qa_df
		self._linked_corpus = linked_corpus

	@property
	def linked_corpus(self) -> Corpus:
		return self._linked_corpus

	@linked_corpus.setter
	def linked_corpus(self, corpus: Corpus):
		raise NotImplementedError("linked_corpus is read-only.")

	def batch_apply(
		self, fn: Callable[[Dict, Any], Awaitable[Dict]], batch_size: int = 32, **kwargs
	) -> "QA":
		qa_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(qa_dict, **kwargs) for qa_dict in qa_dicts]
		results = loop.run_until_complete(process_batch(tasks, batch_size))

		# Experimental feature
		if fn.__name__ == "multiple_queries_gen":
			return self._process_multiple_queries_gen(results)

		return QA(pd.DataFrame(results), self.linked_corpus)

	def batch_filter(
		self, fn: Callable[[Dict, Any], Awaitable[bool]], batch_size: int = 32, **kwargs
	) -> "QA":
		qa_dicts = self.data.to_dict(orient="records")
		loop = get_event_loop()
		tasks = [fn(qa_dict, **kwargs) for qa_dict in qa_dicts]
		masks = loop.run_until_complete(process_batch(tasks, batch_size))
		return QA(self.data[masks], self.linked_corpus)

	def filter(self, fn: Callable[[Dict, Any], bool], **kwargs) -> "QA":
		qa_dicts = self.data.to_dict(orient="records")
		masks = [fn(qa_dict, **kwargs) for qa_dict in qa_dicts]
		return QA(self.data[masks], self.linked_corpus)

	def map(self, fn: Callable[[pd.DataFrame, Any], pd.DataFrame], **kwargs) -> "QA":
		return QA(fn(self.data, **kwargs), self.linked_corpus)

	def make_retrieval_gt_contents(self) -> "QA":
		"""
		Make retrieval_gt_contents column from retrieval_gt column.
		:return: The QA instance that has a retrieval_gt_contents column.
		"""
		self.data["retrieval_gt_contents"] = self.data["retrieval_gt"].apply(
			lambda x: fetch_contents(self.linked_corpus.data, x)
		)
		return self

	def to_parquet(self, qa_save_path: str, corpus_save_path: str):
		"""
		Save the qa and corpus to the AutoRAG compatible parquet file.
		It is not for the data creation, for running AutoRAG.
		If you want to save it directly, use the below code.
		`qa.data.to_parquet(save_path)`

		:param qa_save_path: The path to save the qa dataset.
		:param corpus_save_path: The path to save the corpus.
		"""
		if not qa_save_path.endswith(".parquet"):
			raise ValueError("save_path must be ended with .parquet")
		if not corpus_save_path.endswith(".parquet"):
			raise ValueError("save_path must be ended with .parquet")
		save_df = self.data[
			["qid", "query", "query_type","query_length","retrieval_gt","retrieval_gt_contents", "generation_gt","generation_gt_length","ai_eval_score","eval_reason"]
		].reset_index(drop=True)
		save_df.to_parquet(qa_save_path)
		self.linked_corpus.to_parquet(corpus_save_path)

	def update_corpus(self, new_corpus: Corpus) -> "QA":
		"""
		Update linked corpus.
		Not just replace linked_corpus to the new Corpus,
		it replaces the whole `retrieval_gt` to the new corpus using `linked_raw`.
		The QA data must have a `retrieval_gt` column.

		:param new_corpus: Corpus that you want to replace.
		    Must have valid `linked_raw` and `raw_id`, `raw_start_idx`, `raw_end_idx` columns.
		:return: The QA instance that updated linked corpus.
		"""
		self.data["evidence_path"] = (
			self.data["retrieval_gt"]
			.apply(
				lambda x: fetch_contents(
					self.linked_corpus.data,
					x,
					column_name="path",
				)
			)
			.tolist()
		)
		self.data["evidence_page"] = self.data["retrieval_gt"].apply(
			lambda x: list(
				map(
					lambda lst: list(map(lambda x: x.get("page", -1), lst)),
					fetch_contents(self.linked_corpus.data, x, column_name="metadata"),
				)
			)
		)
		if "evidence_start_end_idx" not in self.data.columns:
			# make evidence start_end_idx
			self.data["evidence_start_end_idx"] = (
				self.data["retrieval_gt"]
				.apply(
					lambda x: fetch_contents(
						self.linked_corpus.data,
						x,
						column_name="start_end_idx",
					)
				)
				.tolist()
			)

		# matching the new corpus with the old corpus
		path_corpus_dict = QA.__make_path_corpus_dict(new_corpus.data)
		new_retrieval_gt = self.data.apply(
			lambda row: QA.__match_index_row(
				row["evidence_start_end_idx"],
				row["evidence_path"],
				row["evidence_page"],
				path_corpus_dict,
			),
			axis=1,
		).tolist()
		new_qa = self.data.copy(deep=True)[["qid", "query", "generation_gt"]]
		new_qa["retrieval_gt"] = new_retrieval_gt
		return QA(new_qa, new_corpus)

	@staticmethod
	def __match_index(target_idx: Tuple[int, int], dst_idx: Tuple[int, int]) -> bool:
		"""
		Check if the target_idx is overlap by the dst_idx.
		"""
		target_start, target_end = target_idx
		dst_start, dst_end = dst_idx
		return (
			dst_start <= target_start <= dst_end or dst_start <= target_end <= dst_end
		)

	@staticmethod
	def __match_index_row(
		evidence_indices: List[List[Tuple[int, int]]],
		evidence_paths: List[List[str]],
		evidence_pages: List[List[int]],
		path_corpus_dict: Dict,
	) -> List[List[str]]:
		"""
		Find the matched passage from new_corpus.

		:param evidence_indices: The evidence indices at the corresponding Raw.
		        Its shape is the same as the retrieval_gt.
		:param evidence_paths: The evidence paths at the corresponding Raw.
		        Its shape is the same as the retrieval_gt.
		:param path_corpus_dict: The key is the path name, and the value is the corpus dataframe that only contains the path in the key.
		        You can make it using `QA.__make_path_corpus_dict`.
		:return:
		"""
		result = []
		for i, idx_list in enumerate(evidence_indices):
			sub_result = []
			for j, idx in enumerate(idx_list):
				path_corpus_df = path_corpus_dict[evidence_paths[i][j]]
				if evidence_pages[i][j] >= 0:
					path_corpus_df = path_corpus_df.loc[
						path_corpus_df["metadata"].apply(lambda x: x.get("page", -1))
						== evidence_pages[i][j]
					]
				matched_corpus = path_corpus_df.loc[
					path_corpus_df["start_end_idx"].apply(
						lambda x: QA.__match_index(idx, x)
					)
				]
				sub_result.extend(matched_corpus["doc_id"].tolist())
			result.append(sub_result)
		return result

	@staticmethod
	def __make_path_corpus_dict(corpus_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
		return {
			path: corpus_df[corpus_df["path"] == path]
			for path in corpus_df["path"].unique()
		}

	# Experimental feature
	def _process_multiple_queries_gen(self, results: List[Dict]) -> "QA":
		data = []
		for result in results:
			queries = result["query"].split("\n")
			for query in queries:
				new_result = {
					key: (str(uuid.uuid4()) if key == "qid" else result[key])
					for key in result.keys()
				}
				new_result["query"] = query
				data.append(new_result)
		df = pd.DataFrame(data)
		return QA(df, self.linked_corpus)

	def add_validation(
			self,
			evaluator: QAEvaluator,
			score_threshold: Optional[int] = None,
			batch_size: int = 20
	) -> "QA":
		"""链式验证方法"""
		# 执行验证
		validated_qa = self.batch_apply(
			cross_validate_qa,
			batch_size=batch_size,
			evaluator=evaluator,
			linked_corpus=self.linked_corpus
		)

		# 应用分数过滤
		if score_threshold is not None:
			return validated_qa.filter(
				lambda x: x.get('ai_eval_score', 0) >= score_threshold
			)
		return validated_qa






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