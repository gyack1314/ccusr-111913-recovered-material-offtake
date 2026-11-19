# 再生料混配报价的质量风险与承销量分析资料契约

以下为项目本地合成资料，不代表真实居民、机构、金额或政策。所有数据为 UTF-8 JSON Lines，每行是一个完整对象；金额采用整数分，null 表示未知或不适用，不能自动等同于零。时间戳含 +08:00 偏移，日期为当地业务日。known_at 或 reported_at 控制资料何时可得，业务发生时间不意味着当时已经获知。

## 业务含义

资料中的moisture为湿基水质量/湿总质量，impurity为干基杂质/干总质量。样本stratum和represented_wet_kg限制代表范围，targeted不拥有随机代表性。工艺试验不是完整配方答案，recipe只标识批准的工艺类别；同trial_lot重复试验不可当独立批次。actuals中processed表示原料已离开可用库存，不可当作客户已经接受的合格货。报价风险评估必须说明场景构造与证据范围，risk_limit只是客户允许上限，样本不足可判断无法支持，而非保证一定可行。日期、金额与质量基础统一后方可比较。

## 原始记录

### lots

| 字段 | 含义 |
| --- | --- |
| id | 原料批次 |
| source | 来源主体 |
| material | 材质 |
| wet_kg | 可见湿重 |
| received_at | 接收时点 |
| forbidden_tags | 已知用途限制 |

### assays

| 字段 | 含义 |
| --- | --- |
| id | 检测身份 |
| lot | 料批 |
| sample_stratum | 取样层 |
| sampling_basis | random_mass随机质量取样、targeted定向取样 |
| represented_wet_kg | 代表质量范围 |
| moisture_fraction | 湿基含水率 |
| impurity_fraction_dry | 干基杂质率 |
| polymer_fraction_dry | 干基聚合物成分，未测为空 |
| known_at | 获知时间 |
| supersedes | 明确替代的检测身份 |

### trials

| 字段 | 含义 |
| --- | --- |
| id | 试验身份 |
| recipe | 配方类别 |
| trial_lot | 独立试验批身份 |
| input_wet_kg | 湿基投入 |
| moisture | 湿基含水率 |
| impurity | 干基杂质率 |
| saleable_dry_kg | 可售干重 |
| byproduct_kg | 副产物湿重 |
| cost_cents | 加工费用 |
| known_at | 资料获知时间 |
| fold | 预留开发或后续核对范围 |

### inquiries

| 字段 | 含义 |
| --- | --- |
| id | 询价身份 |
| material | 材质 |
| dry_kg | 需求干重 |
| minimum_polymer | 最低干基成分，非适用为空 |
| max_impurity | 最高干基杂质 |
| deadline | 最晚交付日 |
| truck_dry_kg | 不可拆交付单位 |
| cents_per_dry_kg | 承购单价 |
| reject_penalty_cents | 拒收损失 |
| banned_sources | 禁止来源 |
| risk_limit | 可接受缺交概率上限 |

### capacity

| 字段 | 含义 |
| --- | --- |
| id | 日能力身份 |
| day | 加工日 |
| wet_kg | 湿基最大加工量 |
| recipes | 可用配方 |
| cost_cents | 基础班次成本 |

### actuals

| 字段 | 含义 |
| --- | --- |
| id | 回执身份 |
| lot | 来源料批 |
| order | 对应承购 |
| kind | 加工或客户复检 |
| quantity_kg | 实际作用数量 |
| observed_impurity | 实测干基杂质，未知为空 |
| known_at | 获知时间 |

