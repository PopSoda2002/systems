from weighted_sum import weighted_sum_fwd
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