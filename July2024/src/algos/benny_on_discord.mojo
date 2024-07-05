from algorithm.functional import vectorize, parallelize
from algorithm import parallel_memcpy


fn calculate_block[
    Type: DType, M: Int, N: Int, K: Int, //, BM: Int, BN: Int
](
    bm: Int,
    bn: Int,
    inout res: Matrix[Type, M, N],
    a: Matrix[Type, M, K],
    b: Matrix[Type, K, N],
):
    var acc = stack_allocation[BM * BN, Type]()
    memset_zero(acc, BM * BN)

    for k in range(K):
        var b = b.data + k * N

        for m in range(BM):
            var a_val = a[bm + m, k]
            var acc = acc + m * BN

            fn inner_n[W: Int](n: Int) capturing:
                SIMD[size=W].store(
                    acc,
                    n,
                    SIMD[size=W]
                    .load(b, bn + n)
                    .fma(a_val, SIMD[size=W].load(acc, n)),
                )

            vectorize[inner_n, simdwidthof[Type](), size=BN]()

    for m in range(BM):
        parallel_memcpy(res.data + (bm + m) * N + bn, acc + m * BN, BN)


fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias TARGET_BLOCK_SIZE_N = 8  # L1 / sizeof[Type]
    alias TARGET_BLOCK_SIZE_M = 4  # L2 / sizeof[Type]

    fn process_block[current_block_size_m: Int](m_index_of_block: Int) capturing:
        @parameter
        for bn in range(0, N if current_block_size_m else 0, TARGET_BLOCK_SIZE_N):
            calculate_block[current_block_size_m, min(TARGET_BLOCK_SIZE_N, N - bn)](
                m_index_of_block * current_block_size_m 
                if current_block_size_m == TARGET_BLOCK_SIZE_M 
                else m_index_of_block, 
                bn, res, a, b
            )

    parallelize[process_block[TARGET_BLOCK_SIZE_M]](M // TARGET_BLOCK_SIZE_M)
    process_block[M % TARGET_BLOCK_SIZE_M](M - M % TARGET_BLOCK_SIZE_M)
