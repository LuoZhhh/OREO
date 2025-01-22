import torch
local_labels = torch.tensor([[0], [1]], dtype=torch.int64)  # 形状 [2, 1]
logits = torch.zeros((2, 512))
bsize = logits.shape[0]
logps_raw = torch.log_softmax(logits, dim=-1)
logps = logps_raw.gather(-1, local_labels.unsqueeze(-1)).squeeze(-1)
ring_attn_size = self.strategy.ring_attn_size
if self.strategy.ring_attn_group is not None:
    logps = all_gather(logps, self.strategy.ring_attn_group)
print(f'Before logps shape: {bsize}, {logps.shape}')
logps_chunks = torch.chunk(logps, ring_attn_size, dim=0)  # list of [bsize, seqlen_i] tensors
lopgs = torch.cat(logps_chunks, dim=1).view(bsize, -1)
print(f'After logps shape: {logps.shape}, {action_masks.shape}')
accumulated_logps = (logps * action_masks).flip(-1).cumsum(-1).flip(-1)