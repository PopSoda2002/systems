from weighted_sum import weighted_sum_fwd, weighted_sum_bwd
import torch
import triton
from einops import rearrange

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        D, output_dims = x.shape[-1], x.shape[:-1]

        input_shape = x.shape
        x = rearrange(x, "... d -> (... d)")
        ctx.save_for_backward(x, weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Weight must be a 1D tensor of shape (D,)"
        assert x.is_cuda and weight.is_cuda, "x and weight must be on CUDA"
        assert x.is_contiguous(), "x must be contiguous"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16
        ctx.ROWS_TILE_SIZE = 16
        ctx.input_shape = input_shape

        y = torch.empty(output_dims, device=x.device)

        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
            x, weight, y, x.stride(0), x.stride(1), weight.stride(0), y.stride(0), n_rows, D, ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE,
        )

        return y.view(input_shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        ROW_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE

        n_rows, D = x.shape

        partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROW_TILE_SIZE), D), device=x.device, dtype=x.dtype)
        grad_x = torch.empty_like(x)

        weighted_sum_bwd[(triton.cdiv(n_rows, ROW_TILE_SIZE),)](
            x, weight,
            grad_output,
            grad_x, partial_grad_weight,
            x.stride(0), x.stride(1), weight.stride(0), grad_output.stride(0), grad_x.stride(0), grad_x.stride(1), partial_grad_weight.stride(0), partial_grad_weight.stride(1), n_rows, D,
            ROW_TILE_SIZE, D_TILE_SIZE,
        )
        grad_weight = partial_grad_weight.sum(axis=0)

        return grad_x, grad_weight
