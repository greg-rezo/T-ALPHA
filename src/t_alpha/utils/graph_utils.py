# The functions in this file are adapted from the E(n) EGNN software corresponding
# to this paper: <https://arxiv.org/abs/2102.09844v3>

import torch


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Sum the data entries for each segment (node). This is useful for aggregating
    edge features for each node.

    The aggregation performed here is equivalent to:
    h_i = Σ_j m_ij
    where m_ij are the edge features (messages passed from node j to node i).

    :param data: Tensor containing data to sum.
    :param segment_ids: Segment indices that define which node each edge belongs to.
    :param num_segments: Total number of segments (nodes).
    :return: Summed data for each segment (node).
    """
    result_shape = (num_segments, data.size(1))  # Shape of the result
    result = data.new_full(result_shape, 0)  # Initialize result tensor with zeros
    segment_ids = segment_ids.unsqueeze(-1).expand(
        -1, data.size(1)
    )  # Expand segment IDs to align dimensions
    result.scatter_add_(
        0, segment_ids, data
    )  # Add data into the corresponding segments
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    """
    Compute the mean of data entries for each segment (node). This is useful for
    aggregating edge features for each node and then averaging them.

    The aggregation performed here is equivalent to:
    h_i = (1 / |N(i)|) Σ_j m_ij
    where N(i) is the set of neighboring nodes and m_ij are the edge features (messages).

    :param data: Tensor containing data to average.
    :param segment_ids: Segment indices that define which node each edge belongs to.
    :param num_segments: Total number of segments (nodes).
    :return: Averaged data for each segment (node).
    """
    result_shape = (num_segments, data.size(1))  # Shape of the result
    segment_ids = segment_ids.unsqueeze(-1).expand(
        -1, data.size(1)
    )  # Expand segment IDs to align dimensions
    result = data.new_full(result_shape, 0)  # Initialize result tensor with zeros
    count = data.new_full(result_shape, 0)  # Initialize count tensor with zeros
    result.scatter_add_(
        0, segment_ids, data
    )  # Sum the data into corresponding segments
    count.scatter_add_(
        0, segment_ids, torch.ones_like(data)
    )  # Count how many times each segment is referenced
    return result / count.clamp(
        min=1
    )  # Compute the mean by dividing the sum by the count
