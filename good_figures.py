# The first edition
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Set the style
# sns.set_style("whitegrid")

# # Data
# models = [
#     "FunReason(7B-xLAM)",
#     "GPT-4o",
#     "ToolACE(8B)",
#     "TooL-N1(7B-xLAM)",
#     "QwQ(32B)",
#     "Qwen2.5(7B-Instruct)",
# ]
# accuracy = [83.66, 82.83, 82.57, 82.01, 80.98, 76.95]

# # Vivid color palette
# colors = [
#     "#FF6B6B",  # Coral Red
#     "#4ECDC4",  # Turquoise
#     "#45B7D1",  # Sky Blue
#     "#FFA07A",  # Light Salmon
#     "#98D8C8",  # Mint Green
#     "#F7DC6F",  # Sunny Yellow
# ]

# # Create figure and axis
# fig, ax = plt.subplots(figsize=(12, 8))

# # Create vertical bar plot with vivid colors
# bars = ax.bar(models, accuracy, color=colors, edgecolor="black", linewidth=1.5)

# # Customize the plot
# ax.set_ylabel("Overall Accuracy", fontsize=14, fontweight="bold")
# ax.set_xlabel("Model", fontsize=14, fontweight="bold")
# ax.set_title(
#     "Performance on BFCLv2 Leaderboard", fontsize=18, fontweight="bold", pad=20
# )

# # Rotate x-axis labels for better readability
# plt.xticks(rotation=45, ha="right")

# # Add value labels on top of bars
# for bar, value in zip(bars, accuracy):
#     height = bar.get_height()
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,
#         height + 0.5,
#         f"{value}%",
#         ha="center",
#         va="bottom",
#         fontsize=12,
#         fontweight="bold",
#     )

# # Customize grid and spines
# ax.grid(axis="y", alpha=0.3)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)


# # Set y-axis limit to give space for labels
# ax.set_ylim(75, 85)  # y-axis range 75-85

# # Adjust layout
# plt.tight_layout()
# plt.savefig("/Users/arac/Desktop/elegant_readme/readme_assets/bfclv2_leaderboard.png")

import matplotlib.pyplot as plt

plt.rcParams["patch.linewidth"] = 0


def funreason_plot():
    # Data
    models = [
        "FunReason(7B-xLAM)",
        "GPT-4o",
        "ToolACE(8B)",
        "TooL-N1(7B-xLAM)",
        "QwQ(32B)",
        "Qwen2.5(7B-Instruct)",
    ]
    accuracy = [83.66, 82.83, 82.57, 82.01, 80.98, 76.95]

    # Build the color list
    # 1. FunReason → Sky Blue
    sky_blue = "#45B7D1"  # DodgerBlue
    colors = [sky_blue]

    # 2 & 3. Remaining bars: dimming turquoise palette that contrasts sharply with sky blue
    turquoises = [
        # "#008B8B",
        # "#008080",
        # "#006666",
        # "#004D4D",
        # "#003333",
        # "#45B7D1",
        # "#333333",
        "#555555",
        "#777777",
        "#999999",
        "#BBBBBB",
        "#DDDDDD",
    ]  # bright → dark turquoise
    colors.extend(turquoises)

    fig, ax = plt.subplots(figsize=(12, 8))

    bars = ax.bar(models, accuracy, color=colors, edgecolor="none", linewidth=1.5)

    ax.set_ylabel("Overall Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_title(
        "Performance on BFCLv2 Leaderboard", fontsize=18, fontweight="bold", pad=20
    )

    # plt.xticks(rotation=45, ha="right")

    for bar, value in zip(bars, accuracy):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.15,
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(75, 85)

    # plt.tight_layout()
    # plt.show()
    plt.savefig("/Users/arac/Desktop/notepad/readme_assets/bfclv2_leaderboard.png")


def deepsearch_plot():
    # Set the style
    # sns.set_style("whitegrid")

    # Data
    models = [
        "RAG-R1-mq",
        "RAG-R1-sq",
        "R1-Researcher",
        "Search-R1",
        "Search-o1",
        "IRCoT",
        # "RAG (Qwen2.5-7B-Instruct)IRCoT",
    ]
    accuracy = [
        0.495,
        0.474,
        0.442,
        0.377,
        0.348,
        0.284,
        # 0.276,
    ]

    # Vivid color palette
    colors = [
        "#45B7D1",
        # "#333333",
        # "#555555",
        "#6EC3D9",
        "#777777",
        "#999999",
        "#BBBBBB",
        "#DDDDDD",
    ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create vertical bar plot with vivid colors
    bars = ax.bar(models, accuracy, color=colors, edgecolor="none", linewidth=1.5)

    # Customize the plot
    ax.set_ylabel("Overall Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_title(
        "Performance on HotpotQA Benchmark", fontsize=18, fontweight="bold", pad=20
    )

    # Rotate x-axis labels for better readability
    # plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for bar, value in zip(bars, accuracy):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.007,
            f"{value}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Customize grid and spines
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set y-axis limit to give space for labels
    ax.set_ylim(0.2, 0.52)  # y-axis range 75-85

    # Adjust layout
    # plt.tight_layout()
    # plt.show()
    plt.savefig("/Users/arac/Desktop/notepad/readme_assets/hotpotqa_benchmark.png")


if __name__ == "__main__":
    funreason_plot()
    deepsearch_plot()
