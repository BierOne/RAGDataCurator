from evaluator import TestEvaluator
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name',
        type=str,
        help="name for the trial directory",
        default=None
    )

    # args for running multiple configs
    parser.add_argument(
        '--config_dir',
        type=str,
        help="dir for loading config files",
        default=None,
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help="Output dir for storing generated single-experiment config files",
        default="/home/lyb/RAG/data/eli5_data/20example"
    )
    parser.add_argument(
        '--project_dir',
        type=str,
        help="Output dir for storing generated logs",
        default="./project"
    )

    # args for single run
    parser.add_argument(
        '--single',
        action='store_true',
        help="only run single config",
    )
    parser.add_argument(
        '--yaml_file',
        type=str,
        help="file path for loading config",
        default=None
    )

    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    config_dir = Path(args.config_dir) if args.config_dir is not None else None
    evaluator = TestEvaluator(qa_data_path=(data_dir/'qa.parquet').as_posix(),
                          corpus_data_path=(data_dir/'corpus.parquet').as_posix(),
                          project_dir=args.project_dir)
    evaluator.init_trial(trial_name=args.name, yaml_dir=config_dir)

    if args.single:
        evaluator.run_single_pass(yaml_file=args.yaml_file,
                                  save_name="final_results.csv")
    else:
        evaluator.run_undone_configs()


if __name__ == '__main__':
    main()

# python ./ragas/run_exp_runner.py -n autorag --yaml_file  ./experiments/ollama_config.yaml --single --data_dir ./ragas/data --project_dir ./ragas/project
# python ./ragas/run_exp_runner.py -n autorag --yaml_file experiments/ollama_config.yaml --single --data_dir ./ragas/data --project_dir ./ragas/project