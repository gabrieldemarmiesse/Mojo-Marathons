from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize

from src.algos import (
    simple_parralellize_on_rows,
    simplest_for_loop,
    baseline_tutorial,
    benny_on_discord,
    benny_optimized_for_small_sizes,
)


fn main() raises:
    # bench_matmul[simplest_for_loop.matmul]("simplest_for_loop")
    # bench_matmul[simple_parralellize_on_rows.matmul]("simple_parralellize_on_rows")
    # bench_matmul[baseline_tutorial.matmul]("baseline_tutorial")
    bench_matmul[benny_on_discord.matmul]("benny_on_discord_8_4_1")
    bench_matmul[benny_optimized_for_small_sizes.matmul]("benny_optimized_for_small_sizes")
