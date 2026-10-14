import json
import re
from collections import defaultdict
import argparse
import csv
from pathlib import Path
from utils import load_json, load_jsonl

DATASET_ORDER = ["arkitscenes", "scannet", "scannetpp"]

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "vsibench_qa.json"
qas = json.load(open(DATA_PATH))
scene_to_dataset = {
    i["scene_name"]: i["dataset"]
    for i in qas
    if i.get("scene_name") and i.get("dataset")
}
arkit_scene_names = set([i["scene_name"] for i in qas if i["dataset"] == "arkitscenes"])
scannet_scene_names = set([i["scene_name"] for i in qas if i["dataset"] == "scannet"])
scannetpp_scene_names = set([i["scene_name"] for i in qas if i["dataset"] == "scannetpp"])


def new_result_bucket():
    return {
        'total': 0,
        'correct': 0,
        'scores': [],
        'empty_pred': 0,
    }


def new_results():
    return defaultdict(new_result_bucket)


def infer_dataset(item):
    if item.get("dataset"):
        return item["dataset"]
    return scene_to_dataset.get(item.get("scene_name"), "unknown")


def item_matches_dataset(item, dataset):
    if dataset in (None, "all"):
        return True
    return infer_dataset(item) == dataset

def normalize_text(text):
    """
    标准化文本：
    1. 转为字符串
    2. 转小写
    3. 移除所有标点符号和空格
    例如: "Turn Left." -> "turnleft"
    """
    if text is None:
        return ""
    # 转字符串并转小写
    s = str(text).lower()
    # 移除所有非字母数字字符（包括标点符号和空格）
    # \W 匹配任何非单词字符 (等价于 [^a-zA-Z0-9_])
    # 我们额外把下划线也去掉，只保留纯字母和数字
    s = re.sub(r'[\W_]+', '', s)
    return s

def extract_answer_from_pred(pred_answer):
    if not pred_answer:
        return None
    
    pred_str = str(pred_answer).strip()
    
    # 优先匹配行首的选项，如 "A."
    pattern1 = re.search(r'^([A-Z])\.', pred_str)
    if pattern1:
        return pattern1.group(1)
    
    # 匹配独立的单个大写字母，如 " A " (但在长句子如 "Turn Right" 中，R不会被单独匹配，因为通常不是全大写)
    # 注意：如果回答是纯文本 "Turn Left"，这里通常不会误匹配到单个字母，
    # 除非文本里恰好有个单独的大写字母。
    pattern2 = re.search(r'\b([A-Z])\b', pred_str)
    if pattern2:
        return pattern2.group(1)
    
    # 如果是纯数字，尝试提取
    numbers = re.findall(r'\d+\.?\d*', pred_str)
    if numbers:
        # 简单的启发式：如果看起来像浮点数或整数，优先返回数字
        # 但要注意不要把 "Scene 1" 里的 1 误认为是答案，这里保留原有逻辑
        try:
            return float(numbers[0])
        except:
            pass
    
    # 如果以上都没匹配到明确的 Label 或 数字，返回原始字符串用于后续文本内容匹配
    return pred_str

def is_numeric(value):
    try:
        float(value)
        return True
    except:
        return False

def calculate_mra(pred, ground_truth, thresholds):
    try:
        pred_val = float(pred)
        gt_val = float(ground_truth)
        
        if gt_val == 0:
            return 1.0 if pred_val == 0 else 0.0
        
        score = 0
        for theta in thresholds:
            relative_error = abs(pred_val - gt_val) / gt_val
            if relative_error < (1 - theta):
                score += 1
        
        return score / len(thresholds)
    except:
        return 0.0

def parse_options(options_list):
    """
    解析选项列表，返回 {Label: Normalized_Content} 的映射
    例如: ["A. Turn Left", "B. Turn Back"] -> {'A': 'turnleft', 'B': 'turnback'}
    """
    mapping = {}
    if not options_list:
        return mapping
        
    for opt in options_list:
        opt_str = str(opt).strip()
        # 匹配 "A. xxx" 或 "A) xxx" 或 "A xxx"
        match = re.match(r'^([A-Z0-9])[\.\)\s]\s*(.*)', opt_str, re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            content = match.group(2)
            mapping[label] = normalize_text(content)
    return mapping

def process_vsibench_single(file_path, dataset, data=None):
    thresholds = [0.5 + 0.05 * i for i in range(10)]
    
    results = new_results()
    
    if data is None:
        if file_path.endswith(".jsonl"):
            data = load_jsonl(file_path)
        elif file_path.endswith(".json"):
            data = load_json(file_path)
    
    for item in data:
        if not item_matches_dataset(item, dataset):
            continue
        # 兼容不同的字段名
        question_type = item.get('question_type', 'unknown') if "question_type" in item else item.get('qa_type', 'unknown')
        ground_truth = str(item.get('ground_truth', '')).strip()
        pred_answer = item.get('pred_answer', '')
        options = item.get('options', []) # 获取选项列表
        
        if question_type.startswith("object_rel_direction"):
            question_type = "object_rel_direction"
        
        results[question_type]['total'] += 1
        
        if not pred_answer or str(pred_answer).strip() == '' or len(str(pred_answer).strip()) >= 1000:
            results[question_type]['empty_pred'] += 1
            results[question_type]['scores'].append(0.0)
            continue
        
        extracted_pred = extract_answer_from_pred(pred_answer)
        
        # 数值类型处理
        if is_numeric(ground_truth):
            mra_score = calculate_mra(extracted_pred, ground_truth, thresholds)
            results[question_type]['scores'].append(mra_score)
            results[question_type]['correct'] += mra_score
        else:
            # 字符串/选项类型处理
            is_correct = False
            
            # 1. 尝试直接 Label 匹配 (例如 GT="A", Pred="A")
            if str(extracted_pred).upper() == ground_truth.upper():
                is_correct = True
            
            # 2. 如果 Label 不匹配，尝试基于 Options 的内容匹配
            # 场景：GT="A", Options=["A. Turn Left"], Pred="turn right" -> Should be False
            # 场景：GT="A", Options=["A. Turn Left"], Pred="Turn Left" -> Should be True
            elif options and len(options) > 0:
                # 解析选项 map: {'A': 'turnleft', 'B': 'turnback'}
                option_map = parse_options(options)
                
                # 获取 GT 对应的文本内容
                gt_label = ground_truth.upper()
                gt_content_norm = option_map.get(gt_label, None)
                
                # 获取 Pred 的标准化文本
                pred_norm = normalize_text(extracted_pred)
                
                # 检查：预测内容 == GT对应的选项内容
                if gt_content_norm and pred_norm == gt_content_norm:
                    is_correct = True
                    
                # 额外检查（反向）：如果模型预测的是 Label (如 "B")，但 extract 没提取好或者逻辑混淆
                # 我们可以检查 pred_norm 是否就是 GT_Label (虽然上面 Check 1 已经覆盖了大部分)
                
            # 3. 最后的兜底：如果没有选项，或者直接文本匹配 (针对非选项题)
            # 例如 GT="yes", Pred="Yes."
            else:
                if normalize_text(extracted_pred) == normalize_text(ground_truth):
                    is_correct = True

            if is_correct:
                results[question_type]['correct'] += 1
                results[question_type]['scores'].append(1.0)
            else:
                results[question_type]['scores'].append(0.0)
    
    return results


def load_result_file(file_path):
    if file_path.endswith(".jsonl"):
        return load_jsonl(file_path)
    if file_path.endswith(".json"):
        return load_json(file_path)
    raise ValueError(f"Unsupported result file extension: {file_path}")


def result_total(results):
    return sum(results[t]['total'] for t in results)


def result_correct(results):
    return sum(results[t]['correct'] for t in results)


def result_empty(results):
    return sum(results[t]['empty_pred'] for t in results)


def print_results_table(results, ordered_types, type_mapping, title=None):
    if title:
        print(f"\n{title}")

    print("\n" + "=" * 120)
    headers = [type_mapping.get(t, t) for t in ordered_types if t in results]
    headers.append('Overall')
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * 120)

    counts = []
    for q_type in ordered_types:
        if q_type in results:
            data = results[q_type]
            counts.append(f"{data['total']:>12}")

    overall_total = result_total(results)
    counts.append(f"{overall_total:>12}")

    empty_counts = []
    for q_type in ordered_types:
        if q_type in results:
            data = results[q_type]
            empty_counts.append(f"{data['empty_pred']:>12}")

    overall_empty = result_empty(results)
    empty_counts.append(f"{overall_empty:>12}")

    print("Total:")
    print(" | ".join(counts))
    print("Empty Pred:")
    print(" | ".join(empty_counts))
    print("-" * 120)

    accuracies = []
    for q_type in ordered_types:
        if q_type in results:
            data = results[q_type]
            acc = data['correct'] / data['total'] if data['total'] > 0 else 0
            accuracies.append(f"{acc:>11.2%}")

    overall_correct = result_correct(results)
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0
    accuracies.append(f"{overall_acc:>11.2%}")

    print(" | ".join(accuracies))
    print("=" * 120)


def build_csv_row(file_name, dataset_name, results, ordered_types):
    row = [file_name, dataset_name]

    for q_type in ordered_types:
        if q_type in results:
            data = results[q_type]
            acc = data['correct'] / data['total'] if data['total'] > 0 else 0
            row.append(f"{acc * 100:.2f}")
        else:
            row.append("0.00")

    overall_total = result_total(results)
    overall_acc = result_correct(results) / overall_total if overall_total > 0 else 0
    row.append(f"{overall_acc * 100:.2f}")

    for q_type in ordered_types:
        if q_type in results:
            row.append(results[q_type]['empty_pred'])
        else:
            row.append(0)

    row.append(result_empty(results))
    row.append(overall_total)

    return row

def process_vsibench(file_paths, dataset):
    type_mapping = {
        'object_counting': 'Obj. Count',
        'object_abs_distance': 'Abs. Dist.',
        'object_size_estimation': 'Obj. Size',
        'room_size_estimation': 'Room Size',
        'object_rel_distance': 'Rel. Dist.',
        'object_rel_direction': 'Rel. Dir.',
        'route_planning': 'Route Plan',
        'obj_appearance_order': 'Appr. Order'
    }
    
    ordered_types = [
        'object_counting',
        'object_abs_distance', 
        'object_size_estimation',
        'room_size_estimation',
        'object_rel_distance',
        'object_rel_direction',
        'route_planning',
        'obj_appearance_order'
    ]
    
    all_results = []
    file_names = []
    all_dataset_results = []
    
    for file_path in file_paths:
        print(f"\nProcessing: {file_path}")
        data = load_result_file(file_path)
        results = process_vsibench_single(file_path, dataset, data=data)
        dataset_results = {
            dataset_name: process_vsibench_single(file_path, dataset_name, data=data)
            for dataset_name in DATASET_ORDER
        }
        all_results.append(results)
        all_dataset_results.append(dataset_results)
        file_names.append(Path(file_path).stem)

        title = f"Selected dataset: {dataset}" if dataset != "all" else "All datasets"
        print_results_table(results, ordered_types, type_mapping, title=title)

        print("\nPer-dataset results:")
        for dataset_name in DATASET_ORDER:
            ds_results = dataset_results[dataset_name]
            if result_total(ds_results) == 0:
                print(f"\n{dataset_name}: no examples")
                continue
            print_results_table(
                ds_results,
                ordered_types,
                type_mapping,
                title=dataset_name,
            )
    
    output_path = "vsibench_combined_results.csv"
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        acc_headers = [type_mapping.get(t, t) for t in ordered_types] + ['Overall']
        empty_headers = [
            f"{type_mapping.get(t, t)} Empty" for t in ordered_types
        ] + ['Overall Empty']
        headers = ['File', 'Dataset'] + acc_headers + empty_headers + ['Total']
        writer.writerow(headers)
        
        for file_name, results, dataset_results in zip(
            file_names,
            all_results,
            all_dataset_results,
        ):
            writer.writerow(build_csv_row(file_name, dataset, results, ordered_types))
            for dataset_name in DATASET_ORDER:
                writer.writerow(
                    build_csv_row(
                        file_name,
                        dataset_name,
                        dataset_results[dataset_name],
                        ordered_types,
                    )
                )
    
    print(f"\nCombined results saved to: {output_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate benchmark results")
    parser.add_argument("--benchmark", "-b", type=str, default="vsibench", choices=["vsibench"])
    parser.add_argument("--files", "-f", type=str, nargs='+', required=True, help="One or more result files")
    parser.add_argument("--dataset", "-d", type=str,default="all", help="Dataset name")
    
    args = parser.parse_args()
    
    process_vsibench(args.files, args.dataset)
