# import gradio as gr
# from gradio_leaderboard import Leaderboard, SelectColumns, ColumnFilter
# from pathlib import Path
# import pandas as pd
# import random

# abs_path = Path(__file__).parent


# TYPES = [
#     "str",
#     "str",
#     "str",
#     "str",
#     "str",
#     "str",
#     "str",
#     "bool",
# ]


# df = pd.read_json(str(abs_path / "models_out.json"))
# # Randomly set True/ False for the "MOE" column
# df["MOE"] = ["1" if random.random() > 0.5 else "2" for _ in range(len(df))]
# df["Flagged"] = [random.random() > 0.5 for _ in range(len(df))]

# with gr.Blocks() as demo:
#     gr.Markdown("""
#     # 🥇 Leaderboard Component
#     """)
#     with gr.Tabs():
#         with gr.Tab("Demo"):
#             Leaderboard(
#                 value=df,
#                 # select_columns=SelectColumns(
#                 #     default_selection=config.ON_LOAD_COLUMNS,
#                 #     cant_deselect=["T", "Model"],
#                 #     label="Select Columns to Display:",
#                 # ),
#                 search_columns=["model"],
#                 # hide_columns=["model_name_for_query", "Model Size"],
#                 filter_columns=[
#                     # "num_tokens",
#                     # ColumnFilter("MOE", type="dropdown",
#                     #              default="1", label="Select MoE"),
#                     ColumnFilter("Flagged", type="boolean", default=False),
#                 ],
#                 datatype=TYPES,
#                 column_widths=["20%", "10%"],
#             )
#         with gr.Tab("Docs"):
#             gr.Markdown((Path(__file__).parent / "docs.md").read_text())

# if __name__ == "__main__":
#     demo.launch()
