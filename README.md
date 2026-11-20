# 再生料混配报价的质量风险与承销量分析

本地、离线的报价审批分析应用。它把合成资料中的库存检测、工艺试验、客户询价与加工能力，
转换成可审批的报价方案：哪些料批给哪一单、哪一天加工、承诺多少干重、质量依据是什么、
挤占了哪一单、在什么风险下仍然兑现不了。

合成资料没有现实居民或企业信息，不能作为真实政策、交易或认证依据。应用不联网，
不接真实交易，也不改写 `data/` 下的原始记录。

## 环境

Python 3.11，依赖固定在 `requirements.txt`（numpy / scipy / pandas 等）。已有相同依赖环境无需重复安装；
新环境执行：

```bash
python3 -m pip install -r requirements.txt
```

## 快速开始

```bash
# 资料体检（摘要、字段、身份、引用）
python3 -m recovery check

# 完整演示：在临时目录生成报价快照、接受、再重判，不写项目 .local
python3 -m recovery demo --orders O1 O5 O7
```

真实使用时，先在本地工作区生成报价快照并审批：

```bash
python3 -m recovery quote --orders O1 O5 O7 --format text
python3 -m recovery accept <快照编号> --orders O1,O5
python3 -m recovery candidates --orders O1 O5 O7
```

## 主要能力

| 问题 | 入口 | 输出 |
| --- | --- | --- |
| 这批料到底什么质量、能不能报价 | `lots` / `context` | 中心估计、覆盖、受限原因、三套质量情景 |
| 混配后能交多少、质量如何 | `candidates` / `quote` | 按整车、按干重的联合计划与方案比较 |
| 接这单会挤掉哪单、代价多少 | `quote` 的“挤占与机会成本”、`gaps` | 同一联合情景下的缺交/质量变化 |
| 报价依据是什么、有效期到哪天 | `quote` 快照 | 料批、检测身份、覆盖、假设、敏感性 |
| 补了检测/接了别单，旧承诺还能兑现吗 | `recheck` | 重新判断、触发原因、当前重算 |
| 客户复检不一致怎么办 | `receipt` + `disposition` | 接受/重检/争议/退货，改变剩余履约与已实现利润 |
| 单批物料去了哪里 | `trace` / `gaps` | 已加工、可用、最近报价占用 |
| 报价与实际差异 | `differences` / `realized` | 证据变化、超计划加工、客户复检处置 |

完整命令：

```text
check  context  lots  candidates  quote  accept  cancel
receipt  assay  disposition  recheck  differences  trace  gaps  realized  demo
```

`python3 -m recovery --help` 查看参数。所有命令支持 `--root`（资料根目录）、
`--local`（工作区）、`--plan-date`（决策日）；`quote`/`recheck` 支持 `--format json`。

## 方法要点

* **干湿基分开**：含水率是湿基；杂质与聚合物是干基。干重 = 湿重 × (1−含水率)；
  干基指标按代表干重加权，不与湿基吨数直接混算。
* **取样代表性**：随机质量取样用于中心估计；定向取样只用于压力情景与审核，
  不因“日期更新”覆盖不相容的取样范围。
* **产出率**：只用开发折试验，按 `trial_lot` 分组交叉验证；留出折仅回测。
  配方差异常小于批间波动，报告不预设配方胜出。
* **联合计划**：料批、加工日、班次成本、整车交付、质量门槛与短交代价在同一个
  混合整数规划里求解；所有候选组合共享同一组蒙特卡洛抽样，不按客户分开最优再相加。
* **不确定性**：资料无法识别的相关性（指标共变、产出共同因子）以敏感性区间呈现，
  不把预测区间说成对每车质量的保证。
* **快照不可改写**：报价证据、假设与价格有效期一经生成即固定；后续检测、接受、
  回执、复检只追加决策并重新判断是否仍能兑现。

详细推导、公式与适用边界见 [方法说明](docs/method.md)；原始字段契约见
[资料契约](docs/input-contract.md)。

## 数据、状态与输出

* `data/`：原始合成资料，只读。`data/manifest.json` 声明摘要与引用。
* `.local/`：本地状态——`state.json`、`decisions.jsonl`（追加式决策日志）、
  `snapshots/`（报价快照）、`appended_assays.jsonl` / `appended_actuals.jsonl`（新增检测与回执）。
* `outputs/`：可选的导出目录（例如 `quote --csv-dir outputs`）。
* 不写入数据库，不需要账号、令牌或外网。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖：

* `test_materials.py`：原始文件摘要、字段、引用；
* `test_core.py`：干湿基、取样替代、覆盖不足、产出率留出折、料批不重复承诺、
  禁用来源、质量不可满足、后接受单挤占旧承诺；
* `test_risk.py`：共同抽样、拒收收益口径、质量裕量与缺交校准、区间不是保证；
* `test_workflow.py`：快照不可改写、重复接受、取消释放、复检处置、重判；
* `test_cli.py`：端到端命令，且 `demo` 不写项目 `.local`。

## 容器

```bash
docker build -t recovery-data .
docker run --rm recovery-data python -m unittest discover -s tests -v
docker run --rm recovery-data python -m recovery demo
```

镜像基础固定摘要；容器内同样只读 `data/`，状态写入挂载的 `.local/` 或临时目录。
