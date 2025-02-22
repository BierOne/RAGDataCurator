from autorag.evaluator import Evaluator


evaluator = Evaluator(qa_data_path='qa.parquet', corpus_data_path='corpus.parquet')
evaluator.start_trial('compact_openai.yaml')

