from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize

from src.algos import simple_parralellize_on_rows, simplest_for_loop



fn main() raises:
    test_matmul[simplest_for_loop.matmul]()
    bench_matmul[simplest_for_loop.matmul]("simplest_for_loop")

    test_matmul[simple_parralellize_on_rows.matmul]()
    bench_matmul[simple_parralellize_on_rows.matmul]("simple_parralellize_on_rows")

