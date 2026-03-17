# from matplotlib import pyplot as plt
# import numpy as np
# import pandas as pd
# import sys
# sys.path.append("..")

# df = pd.read_csv("../data/sin_data.csv")

# train_X = df["ymd"].to_numpy().reshape(-1, 1)
# train_T = df["Stock_1"].to_numpy()

# test_X = df["ymd"].to_numpy().reshape(-1, 1)
# test_T = df["Stock_1"].to_numpy()

# print(df.describe())
# print(df.info())
# print(df.shape) # (365, 6)
# print(df.columns)

# fig_col = 3
# fig_index = 2
# fig_num = 1
# col_num = 1

# fig = plt.figure(figsize=(12, 8))

# for i in range(1, fig_index + 1):
#     for c in range(1, fig_col + 1):

#         if fig_num >= (fig_col * fig_index):
#             break
#         ax1 = fig.add_subplot(fig_index, fig_col, fig_num)
#         ax1.set_title(f"{df.columns[col_num]}")
#         ax1.set_xlabel(f"{df.columns[0]}")
#         ax1.set_ylabel(f"{df.columns[col_num]}")
#         ax1.plot(df["ymd"], df.iloc[:, col_num])

#         fig_num += 1
#         col_num += 1

# plt.savefig("original.png")