from autorag.parser import Parser
import multiprocessing
from autorag.chunker import Chunker
from autorag.data import sentence_splitter_modules, LazyInit

'''use one of them to do parsing and chunking operation, remember to adjust data path if necessary'''


# if __name__ == '__main__':
#     multiprocessing.freeze_support()
#     parser = Parser(data_path_glob="all_rawdata.csv",project_dir="source_data//processed_data")
#     print("This is a worker process.")
#     parser.start_parsing("config//parse//Parsing_config.yaml")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    chunker = Chunker.from_parquet(parsed_data_path="source_data//processed_data//parsed_result.parquet",project_dir="3")
    print("This is a worker process.")
    chunker.start_chunking("config//chunk//Chunking_config.yaml")


