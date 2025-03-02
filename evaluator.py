import os
import glob
import json
import logging
import os
import shutil
from datetime import datetime
from itertools import chain
from typing import List, Dict, Optional, Union
from pathlib import Path
import traceback

import pandas as pd
import yaml
import glob

from autorag.nodes.retrieval.base import get_bm25_pkl_name
from autorag.nodes.retrieval.bm25 import bm25_ingest
from autorag.schema import Node
from autorag.nodes.retrieval.vectordb import (
	vectordb_ingest,
	filter_exist_ids,
	filter_exist_ids_from_retrieval_gt,
)
from autorag.utils import (
    cast_qa_dataset,
    cast_corpus_dataset,
    validate_qa_from_corpus_dataset,
)
from autorag.utils.util import (
    load_yaml_config,
    load_summary_file,
    get_event_loop
)
from autorag.vectordb import load_all_vectordb_from_yaml
from autorag.evaluator import Evaluator

from runner import AdvancedRunner as Runner

# for run_generator_node
from autorag.evaluation.util import cast_metrics
from autorag.strategy import measure_speed, filter_by_threshold, select_best
import sys
import io
logger = logging.getLogger("AutoRAG")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='gb18030')  

import portalocker  # Use portalocker for Windows
def write_summary(df_path, df: pd.DataFrame):
    """Safely appends a DataFrame to summary.csv using file locking."""
    file_exists = os.path.isfile(df_path)  # Check if file exists

    with open(df_path, "a") as f:
        try:
            portalocker.lock(f, portalocker.LOCK_EX)  # Lock the file for exclusive access
            df.to_csv(f, header=not file_exists, index=False, encoding='utf-8')  # Append new rows without headers
            f.flush()  # Ensure data is written immediately
        finally:
            portalocker.unlock(f)  # Unlock the file


class TestEvaluator(Evaluator):
    def __init__(self, qa_data_path: str, corpus_data_path: str, project_dir: Optional[str] = None):
        super().__init__(qa_data_path, corpus_data_path, project_dir)
        self.strategy = None
        self.runner = None
        self.trial_path = None
        self.undone_path = None
        self.done_path = None

    def init_trial(self, trial_name: str = None, yaml_dir: str = None):
        """
        Initialize a trial dir with done_configs and undone_configs.
        :param trial_name: The name of the trial.
        :param yaml_dir: The directory of the YAML files.
        """
        def get_new_trial_name() -> str:
            trial_json_path = os.path.join(self.project_dir, "trial.json")
            if not os.path.exists(trial_json_path):
                return "0"
            with open(trial_json_path, "r") as f:
                trial_json = json.load(f)
            return str(int(trial_json[-1]["trial_name"]) + 1)

        def make_trial_dir(trial_name: str):
            trial_json_path = os.path.join(self.project_dir, "trial.json")
            if os.path.exists(trial_json_path):
                with open(trial_json_path, "r") as f:
                    trial_json = json.load(f)
            else:
                trial_json = []

            trial_json.append(
                {
                    "trial_name": trial_name,
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            os.makedirs(os.path.join(self.project_dir, trial_name), exist_ok=True)
            with open(trial_json_path, "w") as f:
                json.dump(trial_json, f, indent=4)

        # Make Resources directory
        os.makedirs(os.path.join(self.project_dir, "resources"), exist_ok=True)
        os.environ["PROJECT_DIR"] = self.project_dir

        if trial_name is None:
            trial_name = get_new_trial_name()
        make_trial_dir(trial_name)
        self.trial_path = os.path.join(self.project_dir, trial_name)
        os.makedirs(os.path.join(self.trial_path, "output"), exist_ok=True)


        if yaml_dir is not None:
            # create config resources
            # trial_path/done_configs: saving configs that have been evaluated
            # trial_path/undone_configs: saving configs that have not been evaluated
            self.done_path = os.path.join(self.trial_path, "done_configs")
            self.undone_path = os.path.join(self.trial_path, "undone_configs")

            if os.path.exists(self.done_path) and os.path.exists(self.undone_path):
                print("Trial directory already exists:", self.trial_path)
                print("Number of done configs:", len(glob.glob(os.path.join(self.done_path, "*.yaml"))))
                print("Number of undone configs:", len(glob.glob(os.path.join(self.undone_path, "*.yaml"))))
            else:
                os.makedirs(self.done_path, exist_ok=False)
                os.makedirs(self.undone_path, exist_ok=False)
                # move yaml files in yaml_dir to undone_configs
                yaml_files = glob.glob(os.path.join(yaml_dir, "*.yaml"))
                print(f"{len(yaml_files)} YAML files found.")
                print("Moving YAML files to the trial directory...")
                for yaml_file in yaml_files:
                    shutil.copy(yaml_file, self.undone_path)



    def init_runner_from_yaml(
        self,
        yaml_path: str,
    ):
        """
        Initialize a Runner object from a YAML configuration file.
        """
        yaml_dict = load_yaml_config(yaml_path)
        vectordb = yaml_dict.get("vectordb", [])

        vectordb_config_path = os.path.join(
            self.project_dir, "resources", "vectordb.yaml"
        )
        with open(vectordb_config_path, "w") as f:
            yaml.safe_dump({"vectordb": vectordb}, f)

        self.__ingest_bm25_full(yaml_dict.get("bm25_tokenizer_list", ['porter_stemmer', 'space']))
        # Ingest VectorDB corpus
        if yaml_dict.get("vectordb", None):
            loop = get_event_loop()
            loop.run_until_complete(self.__ingest_vectordb(yaml_path, full_ingest=True))

        self.runner = Runner.from_yaml(yaml_path, project_dir=self.project_dir)
        self.strategy = yaml_dict.get("strategies", {'metrics': ['meteor', 'rouge', 'bert_score']})


    def __ingest_bm25_full(self, bm25_tokenizer_list: List[str] = []):
        if len(bm25_tokenizer_list) == 0:
            bm25_tokenizer_list = ["porter_stemmer"]
        for bm25_tokenizer in bm25_tokenizer_list:
            bm25_dir = os.path.join(
                self.project_dir, "resources", get_bm25_pkl_name(bm25_tokenizer)
            )
            if not os.path.exists(os.path.dirname(bm25_dir)):
                os.makedirs(os.path.dirname(bm25_dir))
            # ingest because bm25 supports update new corpus data
            bm25_ingest(bm25_dir, self.corpus_data, bm25_tokenizer=bm25_tokenizer)
        print("BM25 corpus embedding complete.")

    async def __ingest_vectordb(self, yaml_path, full_ingest: bool):
        vectordb_list = load_all_vectordb_from_yaml(yaml_path, self.project_dir)
        if full_ingest is True:
            # get the target ingest corpus from the whole corpus
            for vectordb in vectordb_list:
                target_corpus = await filter_exist_ids(vectordb, self.corpus_data)
                await vectordb_ingest(vectordb, target_corpus)
        else:
            # get the target ingest corpus from the retrieval gt only
            for vectordb in vectordb_list:
                target_corpus = await filter_exist_ids_from_retrieval_gt(
                    vectordb, self.qa_data, self.corpus_data
                )
                await vectordb_ingest(vectordb, target_corpus)


    def run(self, prompt: str) -> pd.DataFrame:
        """
        Run the evaluator with a prompt.
        """
        return self.runner.run(prompt)


    def run_with_qa_eval(self, qa_data: pd.DataFrame = None, yaml_name="./0.yaml", save_name=None) -> pd.DataFrame:
        if qa_data is None:
            qa_data = self.qa_data
        metric_names, metric_params = cast_metrics(self.strategy.get("metrics"))
        if metric_names is None or len(metric_names) <= 0:
            raise ValueError("You must at least one metrics for generator evaluation.")
        """
        @result_to_dataframe(["generated_texts", "generated_tokens", "generated_log_probs"])
		pure (generator): A tuple of three elements.
			The first element is a list of a generated text.
			The second element is a list of generated text's token ids, used tokenizer is GPT2Tokenizer.
			The third element is a list of generated text's pseudo log probs.
        """
        results, execution_times = measure_speed(
            func=self.runner.run_with_qa,
            qa_data=qa_data,
            strategy=self.strategy,
        )
        # save results to results.csv
        if save_name is None:
            save_name = yaml_name.split(".")[0] + ".csv"
        write_summary(os.path.join(self.trial_path, "output", save_name), results)

        # get average token usage & time
        average_time = execution_times / len(results)
        token_usages = results["generated_tokens"].apply(len).mean()

        # save scores to summary
        summary_df = pd.DataFrame(
            {
                "filename": [yaml_name],
                "qa_data_dir": [self.qa_data_path],
                "corpus_data_dir": [self.corpus_data_path],
                "execution_time": [average_time],
                "average_output_token": [token_usages],
                **{metric: [results[metric].mean()] for metric in metric_names},
            }
        )
        logger.info(summary_df)
        logger.info("Evaluation complete.")
        return summary_df


    def run_undone_configs(self):
        """
        Run all configurations in the undone_configs directory.
        """
        summary_path = os.path.join(self.trial_path, "summary.csv")
        yaml_files = glob.glob(os.path.join(self.undone_path, "*.yaml"))
        for yaml_file in yaml_files:
            yaml_name = os.path.basename(yaml_file)
            logger.info(f"Running {yaml_file}...")
            # rename the yaml file to locked file
            yaml_file_temp = yaml_file + ".lock"
            try:
                os.rename(yaml_file, yaml_file_temp)
                self.init_runner_from_yaml(yaml_file_temp)
                summary_df = self.run_with_qa_eval(yaml_name=yaml_name)
                write_summary(summary_path, summary_df)

                shutil.move(yaml_file_temp, os.path.join(self.done_path, yaml_name))
                print("Moving", yaml_file, "to", self.done_path)

            except BlockingIOError:
                print(f"Skipping {yaml_file_temp}, already locked by another process.")
                return  # Skip processing if another process is using the file

            except FileNotFoundError:
                print(f"File {yaml_file_temp} not found. It may have been moved or deleted.")
                return  # Prevents errors if file disappears during execution

            except Exception as e:
                # print place of error in the log
                logger.error(f"Error in {yaml_file}:\n{traceback.format_exc()}")
                # recover the name of yaml file
                os.rename(yaml_file_temp, yaml_file)
                break

    def run_single_pass(self, yaml_file: str, save_name="final_results.csv"):
        """
        Run specific configuration (for example, the final configuration) and save the results.
        """
        summary_path = os.path.join(self.trial_path, "summary.csv")
        self.init_runner_from_yaml(yaml_file)

        yaml_name = os.path.basename(yaml_file)
        summary_df = self.run_with_qa_eval(yaml_name="final_"+yaml_name, save_name=save_name)
        write_summary(summary_path, summary_df)



