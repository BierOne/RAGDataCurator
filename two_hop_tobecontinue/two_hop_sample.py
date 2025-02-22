import uuid
from typing import Iterable

import pandas as pd


def random_two_hop(corpus_df: pd.DataFrame, n: int, random_state: int = 30) -> pd.DataFrame:
	# 从 corpus_df 中采样 2*n 条记录
	sample_df = corpus_df.sample(n * 2, random_state=random_state,replace=True)
	# 获取采样后的 doc_id 列表
	doc_ids = sample_df["doc_id"].tolist()

	# 按照每两条组合成一组
	retrieval_list = [
		[[doc_ids[i]], [doc_ids[i + 1]]]  # 每个 QA 条目包含两个 passage，每个 passage 用一个单元素列表表示
		for i in range(0, len(doc_ids), 2)
	]

	# 生成一个 DataFrame，每行一个 QA 数据，包含随机生成的 qid 和组合好的 retrieval_gt 列
	return pd.DataFrame({
		"qid": [str(uuid.uuid4()) for _ in range(n)],
		"retrieval_gt": retrieval_list
	})