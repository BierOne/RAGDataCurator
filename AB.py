import pandas as pd
import logging
from pathlib import Path


def configure_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("data_filter.log"),
            logging.StreamHandler()
        ]
    )


def validate_data(df):
    """验证数据格式"""
    required_columns = {'query', 'generation_gt', 'ai_eval_score', 'quality_tier'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"缺失必要字段: {missing}")

    valid_tiers = {'A', 'B', 'C', 'D', 'E'}
    invalid_tiers = set(df['quality_tier']) - valid_tiers
    if invalid_tiers:
        raise ValueError(f"发现无效质量等级: {invalid_tiers}")


def filter_quality_data(input_path, output_dir="filtered_data"):
    """
    主过滤函数
    :param input_path: 输入文件路径
    :param output_dir: 输出目录
    :return: 过滤后的DataFrame
    """
    try:
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 读取数据
        logging.info(f"正在读取数据: {input_path}")
        df = pd.read_parquet(input_path)
        original_count = len(df)

        # 数据验证
        validate_data(df)

        # 执行过滤
        logging.info("执行质量过滤 (保留A/B等级)...")
        filtered_df = df[df['quality_tier'].isin(['A', 'B'])]
        filtered_count = len(filtered_df)

        # 生成统计信息
        tier_dist = df['quality_tier'].value_counts().to_dict()
        retention_rate = filtered_count / original_count * 100

        # 保存结果
        output_path = Path(output_dir) / f"filtered_{Path(input_path).name}"
        filtered_df.to_parquet(output_path)

        # 记录日志
        logging.info(f"原始数据量: {original_count}")
        logging.info(f"过滤后数据量: {filtered_count} ({retention_rate:.1f}% 保留率)")
        logging.info("质量分布统计:")
        for tier, count in tier_dist.items():
            logging.info(f"- {tier}级: {count}条 ({(count / original_count) * 100:.1f}%)")
        logging.info(f"结果已保存至: {output_path}")

        return filtered_df

    except FileNotFoundError:
        logging.error("输入文件不存在，请检查路径")
    except pd.errors.ParserError:
        logging.error("文件解析错误，请确认文件格式正确")
    except Exception as e:
        logging.error(f"未知错误: {str(e)}")
        raise


if __name__ == "__main__":
    configure_logging()

    # 配置参数
    input_file = "./s256_48_100/pc3/final_qa_evaluated.parquet"

    # 执行过滤
    filtered_data = filter_quality_data(input_file)

    # 示例输出结果
    if filtered_data is not None:
        logging.info("\n过滤后数据示例:")
        logging.info(filtered_data[['query', 'ai_eval_score', 'quality_tier']].head().to_string())
