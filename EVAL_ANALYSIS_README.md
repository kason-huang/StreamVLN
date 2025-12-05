# StreamVLN评估结果分析脚本

## 使用方法

### 基本用法
```bash
python eval_result_analyze.py --result_file results/vals/_unseen/streamvln_3/result.json
```

### 自定义用法
```bash
python eval_result_analyze.py \
    --result_file results/vals/_unseen/streamvln_3/result.json \
    --model_config "4-bit quantization, 16 frames, 8 history, 196 token limit"
```

## 参数说明

- `--result_file`: 结果文件路径 (默认: `results/vals/_unseen/streamvln_3/result.json`)
- `--model_config`: 模型配置描述 (默认: "4-bit quantization, 16 frames, 8 history, 196 token limit")

## 输出内容

脚本会生成以下分析内容：

### 1. 主要性能指标
- 总episode数量
- 成功率 (Success Rate)
- 平均SPL
- 平均OS (Oracle Success)
- 平均导航误差 (Navigation Error)
- 平均步数

### 2. 成功vs失败案例对比
- 成功案例的数量、百分比、平均导航误差、SPL、OS、步数
- 失败案例的相应统计信息

### 3. 场景性能统计
- 每个场景的episode数量、成功率、SPL、OS、导航误差、步数
- 按成功率排序显示

### 4. 极值案例分析
- 最佳SPL案例
- 最差SPL案例
- 最大导航误差案例
- 最小导航误差案例

### 5. 步数分析
- 平均步数和范围
- 达到步数限制的episode统计

### 6. 接近目标分析
- 接近目标的失败案例(误差<3m)统计

### 7. 性能分析总结
- 整体性能评估
- 关键发现和建议

## 示例输出

```
=== StreamVLN模型评估结果详细分析 ===
结果文件: results/vals/_unseen/streamvln_3/result.json
模型配置: 4-bit quantization, 16 frames, 8 history, 196 token limit

主要性能指标:
  总episode数量: 1840
  成功率: 31.85%
  平均SPL: 0.2810
  平均OS: 0.4016
  平均导航误差: 6.7549 meters
  平均步数: 75.8

成功案例分析:
  数量: 586 (31.8%)
  平均导航误差: 1.39m
  平均SPL: 0.8822
  平均OS: 1.0000
  平均步数: 59.9
  导航误差范围: 0.00m - 2.99m

... (更多分析内容)

性能分析总结:
1. 模型成功率约为31.85%，在VLN任务中属于良好水平
2. 成功案例的导航误差明显低于失败案例 (1.39m vs 9.27m)
3. 成功案例平均步数较少，表明路径更直接
4. 不同场景间性能差异较大
5. 平均OS为0.4016，表明模型选择的路径方向基本正确
6. 有2个失败案例接近目标(误差<3m)，说明停止时机判断需要改进
```

## 注意事项

1. 确保结果文件路径正确且文件存在
2. 结果文件应为JSON Lines格式，每行一个JSON对象
3. JSON对象应包含以下字段：scene_id, episode_id, success, spl, os, ne, steps, episode_instruction