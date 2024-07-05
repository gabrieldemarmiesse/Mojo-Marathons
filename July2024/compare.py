# read the two csvs and show the percentage of the difference elementwise
# in a plot. The csvs look like this
# M, N, K, int8, int16, float64
# 1, 47, 97, 4.6200200104135822, 4.6318902954519494, 4.6396641946317487
# 53, 1, 101, 4.1986593942933359, 4.6241494424004781, 2.9351489023308091
# 1024, 512, 256, 4.5209978380307936, 4.5051698723161149, 4.4846903263814362
# 1024, 1024, 512, 4.5770569940968624, 4.6373641069215221, 4.429498041861625


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.colors


def get_dataframes_with_identical_columns():
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    df1 = df1.drop(cols1 - cols2, axis=1)
    df2 = df2.drop(cols2 - cols1, axis=1)
    return df1, df2


def main():
    df1, df2 = get_dataframes_with_identical_columns()

    df = (df2 - df1) * 100 / df1

    # now we plot the percentage difference
    # the x axis is the column names (data types) and the y axis is the row names (M, N, K)
    df_without_mnk = df.drop(["M", " N", " K"], axis=1)
    color_limit = 50
    norm = matplotlib.colors.TwoSlopeNorm(
        vmin=-color_limit, vcenter=0, vmax=color_limit
    )
    plt.imshow(
        df_without_mnk.values, cmap="RdYlGn", interpolation="nearest", norm=norm
    )
    plt.xticks(range(len(df_without_mnk.columns)), df_without_mnk.columns)
    plt.yticks(
        range(len(df.index)),
        [
            f"{df1['M'][i]}, {df1[' N'][i]}, {df1[' K'][i]}"
            for i in range(len(df.index))
        ],
    )
    plt.colorbar()

    # Annotate each cell with the numeric value
    for i in range(len(df_without_mnk)):
        for j in range(len(df_without_mnk.columns)):
            plt.text(
                j,
                i,
                f"{df_without_mnk.values[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
            )

    plt.title(
        f'Improvements {sys.argv[1].removesuffix(".csv")} ->'
        f' {sys.argv[2].removesuffix(".csv")}'
    )

    plt.show()


main()
