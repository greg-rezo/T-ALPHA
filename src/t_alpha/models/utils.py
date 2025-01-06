from torch import nn


class AttentionPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionPooling, self).__init__()
        self.attn = nn.Linear(input_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: [batch, N, input_dim]
        attn_weights = self.softmax(self.attn(x))  # [batch, N, 1]
        weighted_x = x * attn_weights  # [batch, N, input_dim]
        summed_x = weighted_x.sum(dim=1)  # [batch, input_dim]
        return self.proj(summed_x)  # [batch, output_dim]


def masked_mean_pool(x, mask):
    # Expand mask to match feature dimensions
    mask = mask.unsqueeze(-1).float()  # [batch_size, max_num_nodes, 1]

    # Apply mask
    masked_x = x * mask  # Zero out padded nodes

    # Sum over nodes
    summed_x = masked_x.sum(dim=1)  # [batch_size, features]

    # Count of valid nodes per graph
    valid_counts = mask.sum(dim=1).clamp(min=1)  # [batch_size, 1]

    # Compute mean
    mean_x = summed_x / valid_counts  # [batch_size, features]

    return mean_x
