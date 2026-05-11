def parse_log_file(log_file_path):
    """
    解析日志文件，提取每个实验的超参数和恶意节点排名。
    
    参数:
        log_file_path (str): 日志文件路径
    
    返回:
        list: 包含每个实验信息的字典列表，每个字典包括超参数和排名数据
    """
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    experiments = []
    current_experiment = None  # 当前实验的超参数
    in_ranking_section = False  # 是否在排名数据部分
    current_ranking_type = None  # 当前排名类型（max 或 mean）
    rankings = {'max': [], 'mean': []}  # 存储 max 和 mean 的排名

    for line in lines:
        line = line.strip()

        # 识别新的实验开始
        if line.startswith("Namespace("):
            if current_experiment:  # 如果已有实验，保存结果
                experiments.append({
                    'hyperparameters': current_experiment,
                    'max_rankings': rankings['node max'],
                    'mean_rankings': rankings['node mean']
                })
            current_experiment = line  # 更新当前实验的超参数
            rankings = {'node max': [], 'node mean': []}  # 重置排名数据
            in_ranking_section = False

        # 识别排名数据开始
        elif line == "==========":
            in_ranking_section = True
            current_ranking_type = None

        # 识别排名类型（max 或 mean）
        elif in_ranking_section and line in ["node max", "node mean"]:
            current_ranking_type = line

        # 收集排名数据（假设是数字）
        elif in_ranking_section and current_ranking_type:
            try:
                _ = line.split()[0]
            except:
                continue
            line = line.replace(_,'').strip()

            # print(line)
            inner = line[1:-1]  # "27, 10.43"

            # 用逗号分隔
            parts = inner.split(',')  # ["27", " 10.43"]

            # 去除空格
            elements = [part.strip() for part in parts]  # ["27", "10.43"]

            # 转换为正确的类型
            try:
                first = int(elements[0])  # 27
            except:
                continue
            rankings[current_ranking_type].append(first)

    # 保存最后一个实验的结果
    if current_experiment:
        experiments.append({
            'hyperparameters': current_experiment,
            'max_rankings': rankings['node max'],
            'mean_rankings': rankings['node mean']
        })

    return experiments

def analyze_experiments(experiments):
    """
    分析实验数据，统计排名在前10的恶意节点数量，并找出最佳实验。
    
    参数:
        experiments (list): 实验数据列表
    
    返回:
        tuple: (所有实验结果, 最佳实验, 最佳实验的 max_top10_count)
    """
    results = []
    max_top10_count = 0  # 记录最大的 max_top10_count
    best_experiment = None  # 记录最佳实验

    for exp in experiments:
        max_rankings = exp['max_rankings']
        mean_rankings = exp['mean_rankings']
        

        # 统计 max 排名在前10的恶意节点数量
        max_top10 = sum(1 for rank in max_rankings if rank <= 30)
        # 统计 mean 排名在前10的恶意节点数量
        mean_top10 = sum(1 for rank in mean_rankings if rank <= 30)

        # 记录当前实验的结果
        result = {
            'hyperparameters': exp['hyperparameters'],
            'max_top10_count': max_top10,
            'mean_top10_count': mean_top10
        }
        results.append(result)



        # 更新最佳实验（以 max_top10_count 为标准）
        if max_top10 > max_top10_count:
            max_top10_count = max_top10
            best_experiment = result

    return results, best_experiment, max_top10_count

# 使用示例
def main():
    log_file_path = 'cadets_different_threshold.log'  # 替换为你的日志文件路径
    experiments = parse_log_file(log_file_path)
    results, best_experiment, max_top10_count = analyze_experiments(experiments)

    # 打印所有实验的结果
    print("所有实验的统计结果：")
    for result in results:
        print(f"超参数: {result['hyperparameters']}")
        print(f"Max Top10 数量: {result['max_top10_count']}")
        print(f"Mean Top10 数量: {result['mean_top10_count']}")
        print("----------")

    # 打印最佳实验
    print("\n最佳实验（Max Top10 数量最多）：")
    print(f"Max Top10 数量: {max_top10_count}")
    print(f"超参数: {best_experiment['hyperparameters']}")

if __name__ == "__main__":
    main()