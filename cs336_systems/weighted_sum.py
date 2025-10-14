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

@triton.jit
def weighted_sum_bwd(
    x_ptr, weight_ptr, grad_output_ptr,
    grad_x_ptr, partial_grad_weight_ptr,
    stride_xr, stride_xd, stride_wd, stride_gr, stride_gxr, stride_gxd, stride_gwb, stride_gwd, NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,):

    row_tile_idx = tl.program_id(0)
    n_row_tiles = tl.num_programs(0)

    grad_output_block_ptr = tl.make_block_ptr(grad_output_ptr, shape=(NUM_ROWS,), strides=(stride_gr,), offsets=(row_tile_idx * ROWS_TILE_SIZE,), block_shape=(ROWS_TILE_SIZE,), order=(0,))

    x_block_ptr = tl.make_block_ptr(x_ptr, shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd), offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), order=(1,0),)

    weight_block_ptr = tl.make_block_ptr(weight_ptr, shape=(D,), strides=(stride_wd,), offsets=(0,), block_shape=(D_TILE_SIZE), order=(0),)

    grad_x_block_ptr = tl.make_block_ptr(grad_x_ptr, shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd), offsets=(row_tile_idx * ROWS_TILE_SIZE, 0), block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE), order=(1,0),)

    partial_grad_weight_block_ptr = tl.make_block_ptr(partial_grad_weight_ptr, shape=(n_row_tiles, D,), strides=(stride_gwb,stride_gwd), offsets=(row_tile_idx,0), block_shape=(1, D_TILE_SIZE), order=(1, 0))

    for i in range(tl.cdiv(D, D_TILE_SIZE)):
        grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero")

        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")
        
        grad_x_row = grad_output[:, None] * weight[None, :]

        tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0,1))

        row = tl.load(x_block_ptr, boundary_check=(0,1), padding_option="zero")
        partial_grad_weight = tl.sum(grad_output[:, None] * row[None, :], axis=0, keepdim=True)
        tl.store(partial_grad_weight_block_ptr, partial_grad_weight, boundary_check=(1,))

        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
        partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
        grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE))
