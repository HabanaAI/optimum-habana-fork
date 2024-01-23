import torch
import torch.nn as nn
try:
    import xentropy_cuda_lib
except Exception as e:
    xentropy_cuda_lib = None
if 'all_gather_into_tensor' not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base

class SoftmaxCrossEntropyLossFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, smoothing=0.0, ignored_index=-100, inplace_backward=False, process_group=None):
        """The forward function for softmax cross entropy loss.

        logits: (batch, vocab_size)
        labels: (batch,)
        If process_group is not None, we're doing Tensor Parallel: each process is responsible for
        one part of the vocab. The loss needs to be aggregated across processes.
        """
        (batch, vocab_size) = logits.shape
        assert labels.shape == (batch,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        ctx.total_classes = world_size * vocab_size
        if world_size == 1:
            (losses, lse) = xentropy_cuda_lib.forward(logits, labels, smoothing)
            losses.masked_fill_(labels == ignored_index, 0)
            labels_local = labels
        else:
            rank = torch.distributed.get_rank(process_group)
            (vocab_start_index, vocab_end_index) = (rank * vocab_size, (rank + 1) * vocab_size)
            labels_mask = (labels < vocab_start_index) | (labels >= vocab_end_index)
            ignored_mask = labels == ignored_index
            labels_local = torch.where(ignored_mask, labels, labels - vocab_start_index)
            (losses, lse_local) = xentropy_cuda_lib.forward(logits, labels_local, smoothing, world_size * vocab_size)
            assert lse_local.shape == (batch,)
            assert losses.shape == (batch,)
            losses.masked_fill_(ignored_mask, 0)
            lse_allgather = torch.empty(world_size, batch, dtype=lse_local.dtype, device=lse_local.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, lse_local.contiguous(), group=process_group)
            handle_losses = torch.distributed.all_reduce(losses, op=torch.distributed.ReduceOp.SUM, group=process_group, async_op=True)
            lse = torch.logsumexp(lse_allgather, dim=0)
            rank_per_sample = torch.div(labels, vocab_size, rounding_mode='floor')
            lse_local = lse_allgather[rank_per_sample, torch.arange(batch, device=lse_allgather.device)]
            handle_losses.wait()
            if smoothing == 0.0:
                losses += lse - lse_local
            else:
                losses += (1 - smoothing) * (lse - lse_local) + smoothing * (lse - lse_allgather.sum(dim=0))
            losses.masked_fill_(ignored_mask, 0)
        ctx.save_for_backward(logits, lse, labels_local)
        ctx.smoothing = smoothing
        ctx.ignored_index = ignored_index
        ctx.inplace_backward = inplace_backward
        return losses

    @staticmethod
    def backward(ctx, grad_loss):
        (logits, lse, labels) = ctx.saved_tensors
        grad_loss = grad_loss.contiguous()
        grad_loss.masked_fill_(labels == ctx.ignored_index, 0)
        grad_logits = xentropy_cuda_lib.backward(grad_loss, logits, lse, labels, ctx.smoothing, ctx.inplace_backward, ctx.total_classes)
        return (grad_logits, None, None, None, None, None, None)

class CrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-100, reduction='mean', label_smoothing=0.0, inplace_backward=False, process_group=None):
        super().__init__()
        if xentropy_cuda_lib is None:
            raise ValueError('xentropy_cuda_lib is None, probably because importing xentropy_cuda_lib failed.')
        if reduction not in ['mean', 'none']:
            raise NotImplementedError("Only support reduction = 'mean' or 'none'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.inplace_backward = inplace_backward
        self.process_group = process_group

    def forward(self, input, target):
        assert input.is_cuda and target.is_cuda
        loss = SoftmaxCrossEntropyLossFn.apply(input, target, self.label_smoothing, self.ignore_index, self.inplace_backward, self.process_group)
        if self.reduction == 'mean':
            return loss.sum() / (target != self.ignore_index).sum()
        else:
            return loss