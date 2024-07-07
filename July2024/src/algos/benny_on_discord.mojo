from algorithm.functional import vectorize, parallelize
from algorithm import parallel_memcpy


fn calculate_block[
    Type: DType,
    M: Int,
    N: Int,
    K: Int, //,
    current_block_size_m: Int,
    current_block_size_n: Int,
](
    start_of_block_m: Int,
    start_of_block_n: Int,
    inout res: Matrix[Type, M, N],
    a: Matrix[Type, M, K],
    b: Matrix[Type, K, N],
):
    var acc = stack_allocation[
        current_block_size_m * current_block_size_n, Type
    ]()
    memset_zero(acc, current_block_size_m * current_block_size_n)

    for k in range(K):
        var left_of_b_at_k = b.get_pointer(k, 0)

        for m in range(current_block_size_m):
            var a_val = a[start_of_block_m + m, k]
            var acc = acc + m * current_block_size_n

            fn inner_n[simd_size: Int](position_within_n: Int) capturing:
                acc.store[width=simd_size](
                    position_within_n,
                    left_of_b_at_k
                    .load[width=simd_size](start_of_block_n + position_within_n)
                    .fma(
                        a_val, acc.load[width=simd_size](position_within_n)
                    ),
                )

            vectorize[inner_n, simdwidthof[Type](), size=current_block_size_n]()

    for m in range(current_block_size_m):
        parallel_memcpy(
            res.get_pointer(start_of_block_m + m, start_of_block_n),
            acc + m * current_block_size_n,
            current_block_size_n,
        )


fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias TARGET_BLOCK_SIZE_N = 8  # L1 / sizeof[Type]
    alias TARGET_BLOCK_SIZE_M = 4  # L2 / sizeof[Type]

    fn process_block_any_size[
        current_block_size_m: Int
    ](start_of_block_m: Int) capturing:
        @parameter
        for start_of_block_n in range(0, N, TARGET_BLOCK_SIZE_N):
            alias current_block_size_n = min(
                TARGET_BLOCK_SIZE_N, N - start_of_block_n
            )
            calculate_block[
                current_block_size_m,
                current_block_size_n,
            ](start_of_block_m, start_of_block_n, res, a, b)

    fn process_block_of_target_size(m_index_of_block: Int) capturing:
        process_block_any_size[TARGET_BLOCK_SIZE_M](
            start_of_block_m=m_index_of_block * TARGET_BLOCK_SIZE_M
        )

    parallelize[process_block_of_target_size](M // TARGET_BLOCK_SIZE_M)

    alias remainder = M % TARGET_BLOCK_SIZE_M

    @parameter
    if remainder:
        process_block_any_size[M % TARGET_BLOCK_SIZE_M](
            start_of_block_m=M - remainder
        )
