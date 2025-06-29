import gradio as gr
from gradio_leaderboard import Leaderboard, ColumnFilter
from pathlib import Path
import pandas as pd
import json
import glob
import os
from datetime import datetime

abs_path = Path(__file__).parent
data_dir = abs_path / "data"

# 测试地点 - 统一为北京地区
LOCATIONS = ["北京"]

# 定义数据类型
TYPES = ["str", "str", "str", "number", "number", "number", "number", "str"]


def get_latest_date():
    """获取最新的数据日期"""
    pattern = str(data_dir / "leaderboard_*_*.json")
    files = glob.glob(pattern)
    if not files:
        return None

    # 从文件名中提取日期
    dates = []
    for file in files:
        basename = os.path.basename(file)
        # 文件名格式: leaderboard_{location}_{date}.json
        parts = basename.split('_')
        if len(parts) >= 3:
            date_part = parts[2].replace('.json', '')
            try:
                dates.append(datetime.strptime(date_part, "%Y-%m-%d"))
            except ValueError:
                continue

    if dates:
        return max(dates).strftime("%Y-%m-%d")
    return None


def load_location_data(location, date):
    """加载指定地区和日期的数据"""
    filename = f"leaderboard_{location}_{date}.json"
    filepath = data_dir / filename

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"data": [], "columns": []}


# 获取最新日期
latest_date = get_latest_date()
if latest_date is None:
    latest_date = "2025-06-29"  # 默认日期

# 加载各地区数据
location_data = {}
for location in LOCATIONS:
    location_data[location] = load_location_data(location, latest_date)

with gr.Blocks(css="""
    .gradio-container {
        max-width: none !important;
        width: 100% !important;
    }
    .wrap {
        max-width: none !important;
    }
    .container {
        max-width: none !important;
    }
    /* Leaderboard样式 */
    .leaderboard {
        width: 100% !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
    }
    /* 表格容器样式 */
    .leaderboard table {
        width: 100% !important;
        table-layout: auto !important;
    }
    .leaderboard .table-wrap {
        width: 100% !important;
        height: auto !important;
        max-height: none !important;
        overflow: visible !important;
    }
    /* 确保表格不会分成多列 */
    .leaderboard .dataframe {
        width: 100% !important;
        display: block !important;
    }
    .leaderboard .dataframe thead,
    .leaderboard .dataframe tbody {
        width: 100% !important;
        display: table !important;
    }
    /* 通过JavaScript动态添加样式类 */
    .performance-excellent {
        background-color: #d4edda !important;
        color: #155724 !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
    }
    .performance-good {
        background-color: #fff3cd !important;
        color: #856404 !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
    }
    .performance-average {
        background-color: #f8d7da !important;
        color: #721c24 !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
    }
    /* 低延迟数值高亮 */
    .low-latency {
        background-color: #d4edda !important;
        color: #155724 !important;
        font-weight: bold !important;
    }
    /* 高TPS数值高亮 */
    .high-tps {
        background-color: #d1ecf1 !important;
        color: #0c5460 !important;
        font-weight: bold !important;
    }
    /* 确保所有Gradio组件都没有固定高度 */
    .gradio-container, .gradio-container > div, .block {
        height: auto !important;
        max-height: none !important;
        width: 100% !important;
    }
    /* 防止元素被分割成多列 */
    .gr-box, .gr-form, .gr-panel {
        width: 100% !important;
        display: block !important;
    }
    /* 修复可能的多列布局问题 */
    * {
        column-count: 1 !important;
        columns: none !important;
    }
""") as demo:
    gr.Markdown(f"""
    # 🥇 LLM Performance Leaderboard
    
    **数据日期**: {latest_date}
    """)

    # 只显示北京地区的数据，不需要Tab分组
    data = location_data["北京"]
    if data["data"]:
        df_location = pd.DataFrame(data["data"])

        # 如果数据中没有性能等级列，则添加
        if 'ttft_performance' not in df_location.columns:
            df_location['ttft_performance'] = df_location['ttft'].apply(
                lambda x: '🟢 优秀' if x < 200 else (
                    '🟡 良好' if x < 400 else '🔴 一般')
            )

        if 'tps_performance' not in df_location.columns:
            df_location['tps_performance'] = df_location['tps'].apply(
                lambda x: '🟢 优秀' if x > 40 else ('🟡 良好' if x > 20 else '🔴 一般')
            )

        # 重新排列列顺序
        column_order = ['model', 'provider', 'location', 'ttft', 'ttft_performance',
                        'tps', 'tps_performance', 'total_time', 'output_tokens', 'status']
        df_display = df_location[column_order]

        # 更新数据类型
        display_types = ["str", "str", "str", "number",
                         "str", "number", "str", "number", "number", "str"]

        Leaderboard(
            value=df_display,
            search_columns=["model", "provider"],
            filter_columns=[
                ColumnFilter("provider", type="dropdown", label="选择提供商"),
                ColumnFilter("status", type="dropdown", label="状态筛选"),
                ColumnFilter("ttft_performance",
                             type="dropdown", label="延迟等级"),
                ColumnFilter("tps_performance",
                             type="dropdown", label="速度等级"),
            ],
            datatype=display_types,
            elem_classes=["leaderboard"],
            height=800,
            interactive=True,
        )
    else:
        gr.Markdown("## 暂无数据")

    # 添加说明信息
    gr.Markdown(f"""
    ## LLM性能评测排行榜说明
    
    **当前数据日期**: {latest_date}  
    **测试地区**: 北京
    
    本排行榜展示了不同LLM模型的性能测试表现，包括：
    
    - **首Token延迟(ms)**: 从请求开始到收到第一个token的时间（越低越好）
    - **生成速度(tokens/s)**: 每秒生成的token数量（越高越好）
    - **总耗时(ms)**: 完成整个响应的总时间（越低越好）
    - **输出Token数**: 生成的token总数
    
    数据来源于实时性能测试，每日更新。
    
    ### 性能等级说明
    - **🟢 优秀**: 延迟 < 200ms, 速度 > 40 tokens/s
    - **🟡 良好**: 延迟 200-400ms, 速度 20-40 tokens/s  
    - **🔴 一般**: 延迟 > 400ms, 速度 < 20 tokens/s
    
    ### 支持的模型提供商
    - **硅基流动**: DeepSeek、Qwen、Yi、Llama、Gemma等
    - **阿里云**: Qwen、DeepSeek
    - **火山引擎**: 豆包、DeepSeek
    - **DeepSeek**: DeepSeek Chat
    - 持续扩展中...
    """)

if __name__ == "__main__":
    demo.launch()
