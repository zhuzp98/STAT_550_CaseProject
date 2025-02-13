# %% [markdown]
# # 骨科康复数据分析工作流
# 基于STAT 550项目R分析报告的Python复现版本

# %% [markdown]
# ## 环境配置
# 首次运行前安装依赖：  
# `pip install pandas numpy seaborn statsmodels plotly scikit-learn`

# %%
# 基础库
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# 统计建模
from statsmodels.formula.api import mixedlm
from scipy import stats

# 交互可视化
import plotly.express as px

print("所有依赖库已就绪")

# %% [markdown]
# ## 1. 数据准备
# ### 1.1 数据导入与清洗

# %%
# 读取数据
try:
    c1 = pd.read_csv("./data/data.csv")
    # 统一列名
    c1 = c1.rename(columns={
        'MRN': 'PatientID',
        'Age at injury': 'Age'
    })
    print(f"成功加载数据，维度：{c1.shape}")
except Exception as e:
    print(f"数据加载失败：{str(e)}")

# 二值化转换
binary_map = {
    'Sex': {'F':0, 'M':1},
    'CAD': {'None':0, 'Yes':1},
    'Revision procedure': {'Removal of device':1, 'None':0}
}

for col, mapping in binary_map.items():
    c1[col] = c1[col].map(mapping).fillna(0).astype(int)

# 数据清洗
time_cols = ['Total_3M', 'Total_6M', 'Total_1Y', 'Total_5Y']
c1 = c1.dropna(subset=time_cols, how='all')  # 保留至少一个时间点数据
c1 = c1[(c1[time_cols] != 0).any(axis=1)]     # 排除全零记录

print(f"清洗后数据维度：{c1.shape}")

# %% [markdown]
# ## 2. 数据重构
# ### 2.1 创建分析数据集

# %%
# 创建复合变量
c1['SubAbuse'] = (c1['Substance abuse'] + c1['Alcohol abuse']).clip(upper=1)
# 转换三列心理疾病指标为数值型
c1['Mental illness'] = c1[['Depression', 'Anxiety disorder', 'Psychosis']].apply(
    pd.to_numeric, errors='coerce'  # 将无效值转为NaN
).max(axis=1)

# 长格式转换后清理 NaN 值
long_df = c1.melt(
    id_vars=['PatientID', 'Age', 'Sex', 'Revision procedure'],
    value_vars=time_cols,
    var_name='period',
    value_name='Total'
)

# 时间段映射
period_map = {'Total_3M':3, 'Total_6M':6, 'Total_1Y':12, 'Total_5Y':60}
long_df['month'] = long_df['period'].map(period_map)

# 删除含有 NaN 值的行
long_df = long_df.dropna(subset=['Total'])

print("长格式数据样例：")
print(long_df.head())

# %% [markdown]
# ## 3. 探索性分析（EDA）
# ### 3.1 数据分布可视化

# %%
plt.figure(figsize=(10,6))
sns.histplot(data=long_df, x='Total', hue='period', element='step', kde=True)
plt.title('不同时期总分分布')
plt.show()

# %%
fig = px.box(long_df, x='period', y='Total', color='Sex',
             title='性别对康复评分的影响')
fig.show()

# %% [markdown]
# ## 4. 混合效应模型
# ### 4.1 模型构建

# %%
model = mixedlm("Total ~ month", data=long_df, 
                groups=long_df["PatientID"]).fit()
print(model.summary())

# %% [markdown]
# ## 5. 统计检验
# ### 5.1 设备移除效果分析

# %%
# 准备对比数据
device_group = long_df.groupby(['PatientID', 'Revision procedure'])['Total'].mean().reset_index()

# 检查每组样本量
n_control = len(device_group[device_group['Revision procedure']==0])
n_removal = len(device_group[device_group['Revision procedure']==1])
print(f"对照组样本量: {n_control}, 设备移除组样本量: {n_removal}")

# 只在样本量足够时进行T检验
if n_control > 0 and n_removal > 0:
    t_stat, p_value = stats.ttest_ind(
        device_group.loc[device_group['Revision procedure']==0, 'Total'],
        device_group.loc[device_group['Revision procedure']==1, 'Total'],
        equal_var=False
    )
    print(f"设备移除组 vs 对照组：\nT统计量: {t_stat:.2f}, P值: {p_value:.4f}")
else:
    print("样本量不足，无法进行统计检验")

# %% [markdown]
# ## 6. 结果导出
# ### 6.1 保存处理后的数据

# %%
long_df.to_csv('processed_data.csv', index=False)
print("数据处理结果已保存")