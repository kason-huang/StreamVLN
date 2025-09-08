import argparse
import json
import math

# Define which metrics to compute for each dataset
data_metrics = {
    'r2r': ['success', 'spl', 'ne', 'os', 'steps', 't_input_preprocess', 't_model_encode_sum', 't_model_input_preprocess', 't_model_long_sum', 't_model_short_sum', 't_env_step_sum', 't_total'],
    'rxr': ['success', 'spl', 'os', 'ndtw']
}

def safe_mean(values):
    # Treat NaN as 0
    return sum(0 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in values) / len(values) if values else 0

def main():
    parser = argparse.ArgumentParser(description='Compute average results from a result json file.')
    parser.add_argument('--input', required=True, help='Path to the result json file')
    parser.add_argument('--dataset', required=True, choices=['r2r', 'rxr'], help='Dataset type')
    args = parser.parse_args()

    # 支持每行一个 JSON 对象的文件格式
    results = []
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception as e:
                    print(f"Warning: skip line due to JSON error: {e}")

    metrics = data_metrics[args.dataset]
    metric_values = {m: [] for m in metrics}

    # Each result is expected to be a dict with metrics as keys
    for item in results:
        for m in metrics:
            v = item.get(m, 0)
            # Treat NaN as 0
            if v is None or (isinstance(v, float) and math.isnan(v)):
                v = 0
            metric_values[m].append(v)

    print(f"Average results for dataset: {args.dataset}")
    for m in metrics:
        avg = safe_mean(metric_values[m])
        print(f"{m}: {avg:.4f}")

if __name__ == '__main__':
    main()
