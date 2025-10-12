import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, output_ptr, x_stride_row, x_stride_dim, weight_stride_dim, output_stride_row, ROWS, D, 
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,):
    row_tile_idx = tl.program_id(0)
     
    x_block_ptr = tl.make_block_ptr(x_ptr, shape=(ROWS, D,), strides=(x_stride_row, x_stride_dim), offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), order=(1,0))

    weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(D,), strides=(weight_stride_dim), offsets=(0), block_shape=(D_TILE_SIZE), order=(0))

    output_block_ptr = tl.make_block_ptr(output_ptr, shape=(ROWS,), strides=(output_stride_row,), offsets=(row_tile_idx * ROWS_TILE_SIZE,), block_shape=(ROWS_TILE_SIZE,), order=(0,))

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)  # output for each row

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        row = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero") 
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")

        # row is (ROWS_TILE_SIZE, D_TILE_SIZE)
        # weight is (D_TILE_SIZE,)
        # output is (ROWS_TILE_SIZE,)

        output += tl.sum(row * weight[None, :], axis=1)

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance(D_TILE_SIZE)

    tl.store(output_block_ptr, output, boundary_check=(0,))
