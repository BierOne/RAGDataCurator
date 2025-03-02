# RAGDataCurator

Data creation is the crucial process for RAG. We construct a data creation pipeline for RAG database and corresponding evaluation method, enabling user build up RAG data with their own preference.

## Basic Concepts

![data_creation_pipeline](source_data\picture\data_creation_pipeline.png)

In this new data creation pipeline, we have three schemas. `Raw`, `QA`, and `Corpus`.

- `Raw`: Raw data after you parsed your documents. You can use this data to create `Corpus` data.
- `QA`: Question and Answer pairs. The main part of the dataset. You have to write a great question and answer pair for evaluating the RAG pipeline accurately.
- `Corpus`: The corpus is the text data that the LLM will use to generate the answer.
You can use the corpus to generate the answer for the question.
You have to make corpus data from your documents using parsing and chunking.

In other words,
`Raw` : parsed data
`Corpus` : chunked data
`QA` : Question & Answer dataset based on the corpus



## Parse and Chunk

As shown in the data creation process, we have to do data preprocessing before data creation, including parsing and chunking.



### Parse

It is a crucial step to parse the raw documents.
Because if the raw documents are not parsed well, the RAG will not be optimized well.

Using only YAML files, you can easily use the various document loaders.

The sample parse pipeline looks like this.

```python
from autorag.parser import Parser

parser = Parser(data_path_glob="your/data/path/*")
parser.start_parsing("your/path/to/parse_config.yaml")
```

#### Run Parse Pipeline

##### 1. Set parser instance

```python
from autorag.parser import Parser

parser = Parser(data_path_glob="your/data/path/*")
```

##### 2. Set YAML file
Here is an example of how to use the `langchain_parse` module.

```yaml
modules:
  - module_type: langchain_parse
    file_type: pdf
    parse_method: pdfminer
```
For other type of raw data, you can chance the yaml file modules.


##### 3. Start parsing

Use `start_parsing` function to start parsing.

```python
parser.start_parsing("your/path/to/parse_config.yaml")
```

##### 4. Check the result

If you set `project_dir` parameter, you can check the result in the project directory.
If not, you can check the result in the current directory.

If the parsing is completed successfully, the following three types of files are created in the `project_dir`.

1. Parsed Result
2. Used YAML file
3. Summary file

### chunk
In this section, we will cover how to chunk parsed result.
It is a crucial step because if the parsed result is not chunked well, the RAG will not be optimized well.
Using only YAML files, you can easily use the various chunk methods.

The sample chunk pipeline looks like this.
```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet(parsed_data_path="your/parsed/data/path")
chunker.start_chunking("your/path/to/chunk_config.yaml")
```

####Run Chunk Pipeline

#####1. Set chunker instance
```python
from autorag.chunker import Chunker

chunker = Chunker.from_parquet(parsed_data_path="your/parsed/data/path")
```

```{admonition} Want to specify project folder?
You can specify project directory with `--project_dir` option or project_dir parameter.
```

##### 2. Set YAML file
Here is an example of how to use the `llama_index_chunk` module.

```yaml
modules:
  - module_type: llama_index_chunk
    chunk_method: [ Token, Sentence ]
    chunk_size: [ 1024, 512 ]
    chunk_overlap: 24
```

##### 3. Start chunking

Use `start_chunking` function to start parsing.

```python
chunker.start_chunking("your/path/to/chunk_config.yaml")
```

##### 4. Check the result

If you set `project_dir` parameter, you can check the result in the project directory.
If not, you can check the result in the current directory.

If the chunking is completed successfully, the following three types of files are created in the `project_dir`.

1. Chunked Result
2. Used YAML file
3. Summary file

For example, if chunking is performed using three chunk methods, the following files are created.
`0.parquet`, `1.parquet`, `2.parquet`, `parse_config.yaml`, `summary.csv`

Finally, in the summary.csv file, you can see information about the chunked result, such as what chunk method was used to chunk it.



## QA data curation

In this section, we will cover how to create QA data.

It is a crucial step to create the good QA data. Because if the QA data is bad, the RAG will not be optimized well.

The sample QA creation pipeline looks like this.

```python
import os
from llama_index.llms.openai import OpenAI
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.generation_gt.llama_index_gen_gt import (make_basic_gen_gt)
from QA_definition import Raw, Corpus,QA,setup_evaluator
from autorag.data.qa.query.llama_gen_query import factoid_query_gen
from autorag.data.qa.sample import random_single_hop
import pandas as pd
import nltk
from creation_function import (add_length_columns, classify_query_type, make_jaccard_dedup_filter)
from tqdm import tqdm

batch_qa = (
    corpus_instance
    .sample(random_single_hop, n=sample_size)
    .map(lambda df: df.reset_index(drop=True), )
    .make_retrieval_gt_contents()
    .batch_apply(factoid_query_gen, llm=query_llm,)   #query generation
    .batch_apply(make_basic_gen_gt, llm=answer_llm,)  #gt generation
    .filter(dontknow_filter_rule_based, lang="en", )  #filter Garbled gt
    .filter(make_jaccard_dedup_filter(threshold=0.7), lang="en", )  #duplication checking and filter
    .batch_apply(add_length_columns,batch_size=64  )  #query and gt length attribute
    .batch_apply(classify_query_type)                 #query type
    .add_validation(evaluator, score_threshold=75)    #cross validation
)
batch_qa.to_parquet('./qa.parquet', './corpus.parquet')
```

### 1. Sample retrieval gt

To create question and answer, you have to sample retrieval gt from the corpus data.
You can get the initial chunk data from the raw data.
And then sample it using the `sample` function.

```python
from autorag.data.qa.sample import random_single_hop
from QA_definition import QA

qa = initial_corpus.sample(random_single_hop, n=3).map(
    lambda df: df.reset_index(drop=True),
)
```

You can change the sample method by changing the function to different functions.
Supported methods are below.

|      Method       |                Description                 |
|:-----------------:|:------------------------------------------:|
| random_single_hop |  Randomly sample one hop from the corpus   |
| range_single_hop  | Sample single hop with range in the corpus |


### 2. Get retrieval gt contents to generate questions

At the first step, you only sample retrieval gt ids. But to generate questions, you have to get the contents of the retrieval gt.
To achieve this, you can use the `make_retrieval_gt_contents` function.

```python
qa = qa.make_retrieval_gt_contents()
```

### 3. Generate queries

Now, you use LLM to generate queries.
In this example, we use the `factoid_query_gen` function to generate factoid questions.

```python
from llama_index.llms.openai import OpenAI
from QA_definition import QA
from autorag.data.qa.query.llama_gen_query import factoid_query_gen

llm = OpenAI()
qa = qa.batch_apply(
    factoid_query_gen,  # query generation
    llm=llm,
)
```


### 4. Generate answers

After generating questions, you have to generate answers (generation gt).

```python
from llama_index.llms.openai import OpenAI
from QA_definition import QA
from autorag.data.qa.generation_gt.llama_index_gen_gt import (
    make_basic_gen_gt,
)

llm = OpenAI()

qa = qa.batch_apply(
    make_basic_gen_gt,  # answer generation (basic)
    llm=llm,
```


### 5. Filtering questions

It is natural that LLM generates some bad questions.
So, it is better you filter some bad questions with classification models or LLM models.

To filtering, we use `filter` method.

```python
from llama_index.llms.openai import OpenAI
from QA_definition import QA
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based

llm = OpenAI()
qa = qa.filter(
    dontknow_filter_rule_based,  # filter don't know
    lang="en",
)
```
Besides, for large scale data generation, it is inevitable that there are many repeating QA data.
So, it is better you filter some repeating questions.

To filtering, we use `filter` method and `make_jaccard_dedup_filter` function.

```python
from creation_function import make_jaccard_dedup_filter
from QA_definition import QA

qa = qa.filter(
    make_jaccard_dedup_filter(threshold=0.7),  
    lang="en",        #duplication checking and filter
)
```

### 6. add column attribute

When processing with QA data, we might wonder some attribute about query and answer, like lenght and type of them.

to add column attribute to the output, we use  `add_length_columns`  and `classify_query_type` function.


```python
from creation_function import (add_length_columns, classify_query_type)
from QA_definition import QA

qa = qa.batch_apply(add_length_columns,batch_size=64)
qa = qa.batch_apply(classify_query_type)

```

### 7. QA data Cross validation

It is natural that LLM generates some wrong ground truth.
So, it is better you filter some wrong answer with classification models or LLM models.

To cross-validate gt, we use `add_validation` method.

```python
from QA_definition import QA,setup_evaluator

evaluator = setup_evaluator(
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

qa = qa.add_validation(evaluator, score_threshold=75) 

```

### 8. Save the QA data

Now you can use the QA data for running AutoRAG.

```python
qa.to_parquet('./qa.parquet', './corpus.parquet')
```