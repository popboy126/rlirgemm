# 离散动作自适应熵系数控制使用说明

## 📖 功能概述

离散动作自适应熵系数控制是一种用于强化学习训练中自动调节探索-利用平衡的机制。该功能特别针对离散动作空间（如`thread_type`、`loop_order`、`packing_type`）设计，能够根据训练过程中的实际熵值和loss比例动态调整熵系数，确保探索充分且训练稳定。

### 主要特性

- **双目标控制**：同时监控原始熵值和entropy_loss_ratio
- **自动调节**：无需手动调整熵系数
- **训练稳定**：防止熵loss过高导致训练不稳定
- **充分探索**：防止熵过低导致过早收敛

---

## 🎯 核心概念

### 1. 熵（Entropy）

**定义**：衡量策略分布的随机性，熵越大表示探索越充分。

对于离散动作（如thread_type有4个选项）：
- 最大熵 = ln(4) ≈ 1.386（均匀分布）
- 最小熵 = 0（确定性策略）

**监控指标**：
- 原始熵值：策略分布的实际熵
- entropy_loss_ratio：熵loss占总loss的比例

### 2. 自适应控制目标

**目标1：原始熵值**
- 最小目标：0.3（低于此值会增加系数）
- 最大目标：1.0（高于此值会降低系数）

**目标2：entropy_loss_ratio**
- 最小目标：0.2（低于此值会增加系数）
- 最大目标：0.5（高于此值会降低系数）

### 3. 调整逻辑

```
如果原始熵 < 0.3：
    增加熵系数（促进探索）

如果 entropy_loss_ratio > 0.5：
    降低熵系数（防止训练不稳定）

如果原始熵在[0.3, 1.0]且entropy_loss_ratio在[0.2, 0.5]：
    保持稳定
```

---

## ⚙️ 配置参数说明

### 必需参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ppo_use_adaptive_ent_coef_disc_thread_type` | string | "none" | 是否启用离散动作自适应熵系数控制，可选值："none"（禁用）、"enabled"（启用） |

### 可选参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `ppo_adaptive_ent_target_entropy_min_disc_thread_type` | float | 0.3 | 目标熵最小值，低于此值会增加系数 |
| `ppo_adaptive_ent_target_entropy_max_disc_thread_type` | float | 1.0 | 目标熵最大值，高于此值会降低系数 |
| `ppo_adaptive_ent_target_loss_ratio_min_disc_thread_type` | float | 0.2 | 目标entropy_loss_ratio最小值 |
| `ppo_adaptive_ent_target_loss_ratio_max_disc_thread_type` | float | 0.5 | 目标entropy_loss_ratio最大值 |
| `ppo_adaptive_ent_learning_rate_disc_thread_type` | float | 0.02 | 熵系数更新学习率 |
| `ppo_adaptive_ent_coef_init_disc_thread_type` | float | 0.15 | 初始熵系数 |
| `ppo_adaptive_ent_coef_min_disc_thread_type` | float | 0.05 | 最小熵系数 |
| `ppo_adaptive_ent_coef_max_disc_thread_type` | float | 0.3 | 最大熵系数 |
| `ppo_adaptive_ent_priority_threshold_disc_thread_type` | float | 0.1 | 熵优先阈值，低于此值时优先处理 |

---

## 📝 使用示例

### 示例1：基本使用

在配置文件的`train_config`部分添加以下参数：

```json
{
  "train_config": {
    "trainer_name": "GemmAdaptiveTrainer",
    "device": "cuda:0",

    // ... 其他参数 ...

    // 启用离散动作自适应熵系数控制
    "ppo_use_adaptive_ent_coef_disc_thread_type": "enabled",

    // 使用默认参数即可
    // 如果不指定，会使用上述默认值
  }
}
```

### 示例2：自定义参数

```json
{
  "train_config": {
    "trainer_name": "GemmAdaptiveTrainer",
    "device": "cuda:0",

    // ... 其他参数 ...

    // 启用离散动作自适应熵系数控制
    "ppo_use_adaptive_ent_coef_disc_thread_type": "enabled",

    // 自定义参数
    "ppo_adaptive_ent_target_entropy_min_disc_thread_type": 0.2,
    "ppo_adaptive_ent_target_entropy_max_disc_thread_type": 0.8,
    "ppo_adaptive_ent_target_loss_ratio_min_disc_thread_type": 0.15,
    "ppo_adaptive_ent_target_loss_ratio_max_disc_thread_type": 0.45,
    "ppo_adaptive_ent_learning_rate_disc_thread_type": 0.025,
    "ppo_adaptive_ent_coef_init_disc_thread_type": 0.12,
    "ppo_adaptive_ent_coef_min_disc_thread_type": 0.03,
    "ppo_adaptive_ent_coef_max_disc_thread_type": 0.25,
    "ppo_adaptive_ent_priority_threshold_disc_thread_type": 0.08
  }
}
```

### 示例3：完整配置示例（基于task16）

```json
{
  "train_config": {
    "trainer_name": "GemmAdaptiveTrainer",
    "device": "cuda:0",

    // 基础熵系数配置
    "ppo_ent_coef": 0.01,
    "ppo_ent_coef_disc_loop_order": 0.03,
    "ppo_ent_coef_disc_packing_type": 0.09,
    "ppo_ent_coef_disc_thread_type": 0.15,  // 初始值
    "ppo_ent_coef_cont": 0.001,

    // 连续动作自适应熵系数控制（如果需要）
    "ppo_use_adaptive_ent_coef_cont_type": "v2",
    "ppo_adaptive_ent_target_ratio_cont": 0.72,
    "ppo_adaptive_ent_target_entropy_loss_ratio_cont": 0.35,

    // 离散动作thread_type自适应熵系数控制（新功能）
    "ppo_use_adaptive_ent_coef_disc_thread_type": "enabled",
    "ppo_adaptive_ent_target_entropy_min_disc_thread_type": 0.3,
    "ppo_adaptive_ent_target_entropy_max_disc_thread_type": 1.0,
    "ppo_adaptive_ent_target_loss_ratio_min_disc_thread_type": 0.2,
    "ppo_adaptive_ent_target_loss_ratio_max_disc_thread_type": 0.5,
    "ppo_adaptive_ent_learning_rate_disc_thread_type": 0.02,
    "ppo_adaptive_ent_coef_init_disc_thread_type": 0.15,
    "ppo_adaptive_ent_coef_min_disc_thread_type": 0.05,
    "ppo_adaptive_ent_coef_max_disc_thread_type": 0.3,
    "ppo_adaptive_ent_priority_threshold_disc_thread_type": 0.1,

    // ... 其他参数 ...
  }
}
```

---

## 🔧 工作原理

### 1. 数据采集

在每个训练步骤中，`GemmPPOPolicy`会记录以下信息：

```python
# 在gemm_policy.py中
self._last_mean_raw_entropy_disc_thread_type = actual_entropy  # thread_type的原始熵值
self._last_mean_entropy_loss_ratio = entropy_loss_ratio  # 熵loss占比
```

### 2. 控制器更新

在`training_fn`回调中，调用自适应控制器更新熵系数：

```python
# 在gemm_adaptive_trainer.py中
def training_fn(self, epoch: int, env_step: int) -> None:
    # 1. 连续动作自适应熵系数控制
    if self.ppo_use_adaptive_ent_coef_type.lower() != "none":
        self._update_entropy_coef_cont(epoch, env_step)

    # 2. 离散动作thread_type自适应熵系数控制
    if self.ppo_use_adaptive_ent_coef_disc_thread_type.lower() != "none":
        self._update_entropy_coef_disc_thread_type(epoch, env_step)
```

### 3. 熵系数计算

控制器根据双目标计算新的熵系数：

```python
# 在AdaptiveEntropyControllerDisc.update中
def update(self, epoch, actual_entropy, entropy_loss_ratio):
    # 1. 基于原始熵值的调整
    if actual_entropy < target_entropy_min:
        # 熵过低，增加系数
        entropy_factor = 1.0 + learning_rate * urgency
    elif actual_entropy > target_entropy_max:
        # 熵过高，降低系数
        entropy_factor = 1.0 - learning_rate * urgency * 0.5

    # 2. 基于entropy_loss_ratio的调整
    if entropy_loss_ratio > target_loss_ratio_max:
        # 占比过高，降低系数
        loss_ratio_factor = 1.0 - learning_rate * urgency
    elif entropy_loss_ratio < target_loss_ratio_min:
        # 占比过低，增加系数
        loss_ratio_factor = 1.0 + learning_rate * urgency

    # 3. 综合调整
    combined_factor = (entropy_factor + loss_ratio_factor) / 2.0

    # 4. 更新熵系数
    new_ent_coef = old_ent_coef * combined_factor
    new_ent_coef = clip(new_ent_coef, ent_coef_min, ent_coef_max)

    return new_ent_coef
```

### 4. 应用到训练

新的熵系数会立即应用到下一次梯度更新：

```python
# 在gemm_policy.py中
ent_loss_disc_thread_type = self._ent_coef_disc_thread_type * raw_ent_loss_disc_thread_type
```

---

## 📊 监控指标

训练过程中，可以通过TensorBoard监控以下指标：

### 原始熵值监控

- **文件**：`raw_ent_loss_disc_thread_type_mean.csv`
- **目标范围**：0.3 - 1.0
- **说明**：thread_type的原始熵值，反映探索程度

### 熵loss监控

- **文件**：`ent_loss_disc_thread_type_mean.csv`
- **计算**：原始熵值 × 熵系数
- **说明**：实际参与loss计算的熵项

### entropy_loss_ratio监控

- **文件**：`entropy_loss_ratio_mean.csv`
- **目标范围**：0.2 - 0.5
- **说明**：熵loss占总loss的比例，反映训练稳定性

### 探索分布监控

从`abnormal_history`文件中查看thread_type的探索分布：

```
Thread Type (1, 2, 1) -> frequency: 0.689
Thread Type (1, 1, 1) -> frequency: 0.161
Thread Type (1, 1, 2) -> frequency: 0.104
Thread Type (2, 1, 1) -> frequency: 0.045
```

---

## 💡 最佳实践

### 1. 参数调优建议

**初期训练**：
- 目标熵最小值：0.3-0.4（确保充分探索）
- 初始熵系数：0.12-0.15（基于task16的经验）
- 学习率：0.02-0.03（适中）

**中期训练**：
- 观察熵值波动，如果持续低于0.3，考虑：
  - 降低目标熵最小值（如0.25）
  - 增加学习率（如0.025）
  - 提高最大熵系数（如0.35）

**后期训练**：
- 如果探索分布合理，可以：
  - 提高目标熵最小值（如0.35）
  - 降低最大熵系数（如0.25）

### 2. 监控与调试

**正常现象**：
- 熵值在0.2-0.8之间波动
- entropy_loss_ratio在0.15-0.6之间
- 探索分布逐渐均衡

**异常现象**：
- 熵值持续低于0.1（探索不足）
- entropy_loss_ratio持续高于0.7（训练不稳定）
- 探索分布极端化（如某个选项占比>85%）

**应对策略**：
- 增加学习率
- 提高最大熵系数
- 调整目标熵范围

### 3. 与固定熵系数的对比

| 场景 | 固定系数 | 自适应系数 |
|------|----------|------------|
| 初期探索 | 可能不足 | 自动增强 |
| 中期收敛 | 可能过早 | 保持探索 |
| 后期优化 | 无法调整 | 自动平衡 |
| 训练稳定性 | 可能不稳定 | 自动调节 |

---

## ❓ 常见问题

### Q1: 什么时候应该启用自适应熵系数控制？

**A**: 以下情况建议启用：
1. 训练初期探索不足（熵值快速降到极低）
2. 探索分布极端化（某个选项占比>80%）
3. 不知道合适的熵系数值
4. 训练环境复杂，需要动态调整

### Q2: 如何判断自适应控制是否有效？

**A**: 观察以下指标：
1. 原始熵值保持在0.2以上
2. entropy_loss_ratio在0.15-0.6之间
3. 探索分布相对均衡
4. 训练过程稳定

### Q3: 自适应控制会影响训练速度吗？

**A**: 影响很小：
1. 计算开销：每个epoch只增加一次简单的数学计算
2. 收敛速度：初期可能稍慢（探索更多），但最终效果更好
3. 总体影响：可忽略不计

### Q4: 可以同时为多个离散动作启用自适应控制吗？

**A**: 目前实现仅支持thread_type。如需扩展到其他离散动作（如loop_order、packing_type），可以参照实现类似的功能。

### Q5: 如果熵系数一直增加到最大值怎么办？

**A**: 这说明：
1. 探索严重不足，可能需要检查：
   - 网络结构是否合理
   - Reward设计是否有问题
   - 环境是否过于复杂
2. 可以尝试：
   - 提高最大熵系数上限（如0.4）
   - 增加学习率（如0.03）
   - 使用其他探索策略（如RND）

### Q6: 与连续动作自适应控制可以同时使用吗？

**A**: 可以！两者独立工作：
- 连续动作：监控高斯分布的熵
- 离散动作：监控离散分布的熵
- 两者可以分别配置不同的参数

---

## 📈 实际案例

### Case 1: Task16的经验

**问题**：
- Task14: thread_type熵系数0.1，探索一般
- Task15: thread_type熵系数0.05，探索严重不足（熵降到0.04）
- Task16: thread_type熵系数0.15，探索改善但熵仍有波动

**解决方案**：
启用自适应熵系数控制：
```json
{
  "ppo_use_adaptive_ent_coef_disc_thread_type": "enabled",
  "ppo_adaptive_ent_coef_init_disc_thread_type": 0.15,
  "ppo_adaptive_ent_coef_min_disc_thread_type": 0.05,
  "ppo_adaptive_ent_coef_max_disc_thread_type": 0.3
}
```

**预期效果**：
- 熵值低于0.3时，系数自动增加
- entropy_loss_ratio高于0.5时，系数自动降低
- 保持探索稳定

---

## 📚 参考资料

1. **源代码**：
   - `exploration/adaptive_entropy_controller.py` - AdaptiveEntropyControllerDisc类
   - `gemm_adaptive_trainer.py` - 控制器创建和更新
   - `gemm_policy.py` - 熵值记录

2. **配置文件**：
   - `kunpeng920_rl_parallel_gemm_train_task16.json` - 基础配置示例

3. **训练数据**：
   - `mt-task16/raw_ent_loss_disc_thread_type_mean.csv` - 原始熵值
   - `mt-task16/ent_loss_disc_thread_type_mean.csv` - 熵loss
   - `mt-task16/entropy_loss_ratio_mean.csv` - loss比例

---

## 🔄 更新历史

- **v1.0** (2025-04-01): 初始版本，实现离散动作自适应熵系数控制
- **v1.1** (2025-04-02): 添加双目标控制（熵值 + loss比例）

---

## 📧 联系方式

如有问题或建议，请联系：
- 作者：huche
- 邮箱：hucheng.lew@qq.com
