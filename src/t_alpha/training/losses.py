import torch


def custom_loss(preds, targets, operators):
    """
    Custom loss function for handling predictions based on operator conditions.

    This function computes a loss based on the relationship between predictions
    and targets, as specified by the `operators` argument. It supports four types
    of relationships:
        - '=' and '~' : Mean squared error loss.
        - '>'         : Penalizes predictions less than or equal to targets.
        - '<'         : Penalizes predictions greater than or equal to targets.

    Args:
        preds (torch.Tensor): Predicted values (shape: [batch_size]).
        targets (torch.Tensor): Target values (shape: [batch_size]).
        operators (list of str): List of operator strings ('=', '~', '>', '<')
                                 defining the relationship between predictions
                                 and targets.

    Returns:
        torch.Tensor: The mean loss computed over the batch.
    """

    # Flatten preds and targets to 1D tensors
    preds = preds.view(-1)
    targets = targets.view(-1)

    # Map operators to numerical codes
    operator_to_code = {"=": 0, "~": 0, ">": 1, "<": 2}
    operator_codes = [operator_to_code[op] for op in operators]
    operator_tensor = torch.tensor(
        operator_codes, dtype=torch.long, device=preds.device
    )

    # Create masks for each operator
    equal_or_tilde_mask = operator_tensor == 0
    greater_mask = operator_tensor == 1
    less_mask = operator_tensor == 2

    # Initialize losses tensor
    losses = torch.zeros_like(preds)

    # Compute losses for '=' and '~' operators
    losses[equal_or_tilde_mask] = (
        preds[equal_or_tilde_mask] - targets[equal_or_tilde_mask]
    ) ** 2

    # Compute losses for '>' operator
    greater_condition = preds <= targets
    greater_mask_condition = greater_mask & greater_condition
    losses[greater_mask_condition] = (
        targets[greater_mask_condition] - preds[greater_mask_condition]
    ) ** 2

    # Compute losses for '<' operator
    less_condition = preds >= targets
    less_mask_condition = less_mask & less_condition
    losses[less_mask_condition] = (
        preds[less_mask_condition] - targets[less_mask_condition]
    ) ** 2

    # Compute mean loss
    loss = torch.mean(losses)
    return loss


# Define custom weighted loss function for fine-tuning
def uncertainty_weighted_loss(preds, targets, weights):
    """
    Computes a weighted mean squared error loss for fine-tuning.

    Parameters:
    - preds: torch.Tensor, predicted values
    - targets: torch.Tensor, target values
    - weights: torch.Tensor, weight for each data point to scale the loss

    Returns:
    - loss: torch.Tensor, computed weighted loss
    """
    # Flatten preds, targets, and weights to 1D tensors
    preds = preds.view(-1)
    targets = targets.view(-1)
    weights = weights.view(-1)

    # Compute squared error
    squared_errors = (preds - targets) ** 2

    # Apply weights to the squared errors
    weighted_errors = squared_errors * weights

    # Compute the mean of weighted errors
    loss = torch.mean(weighted_errors)

    return loss
