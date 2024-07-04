from src import Matrix, test_matmul, bench_matmul
from algorithm.functional import vectorize, parallelize


fn matmul_simplest_vesion[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    for m in range(M):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]



fn matmul_parralell[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    
    @parameter
    fn calc_row(m: Int):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]

    parallelize[calc_row](M, M)



fn main() raises:
    test_matmul[matmul_simplest_vesion]()
    bench_matmul[matmul_simplest_vesion]("simplest_version")

    test_matmul[matmul_parralell]()
    bench_matmul[matmul_parralell]("multithreaded_on_columns")

