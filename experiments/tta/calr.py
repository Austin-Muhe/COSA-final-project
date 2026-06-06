import math
from collections import deque


class CALR:
    """Cosine adaptive learning-rate scheduler matching the original COSA logic."""

    def __init__(
        self,
        base_lr: float = 0.001,
        min_lr: float = 0.0001,
        max_lr: float = 0.005,
        adapt_steps: int = 3,
        per_batch_lr_reset: bool = True,
    ):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adapt_steps = max(1, adapt_steps)
        self.per_batch_lr_reset = per_batch_lr_reset
        self.loss_history = deque(maxlen=5)
        self.current_lr = base_lr

    def reset_batch(self):
        if self.per_batch_lr_reset:
            self.current_lr = self.base_lr
            self.loss_history.clear()

    def step(self, loss: float, step: int) -> float:
        if step == 0 and self.per_batch_lr_reset:
            self.current_lr = self.base_lr
            self.loss_history.append(float(loss))
            return self.current_lr

        self.loss_history.append(float(loss))
        recent_losses = list(self.loss_history)[-3:]

        if len(recent_losses) >= 2:
            loss_trend = recent_losses[-1] - recent_losses[0]
            mean_loss = sum(recent_losses) / len(recent_losses)
            loss_variance = sum((value - mean_loss) ** 2 for value in recent_losses) / len(recent_losses)

            if loss_trend > 0 and loss_variance < 1e-6:
                self.current_lr = min(self.current_lr * 1.2, self.max_lr)
            elif loss_trend < -0.01:
                self.current_lr = min(self.current_lr * 1.05, self.max_lr)
            elif abs(loss_trend) < 1e-6:
                self.current_lr = max(self.current_lr * 0.8, self.min_lr)

        if step >= 1:
            cosine_factor = 0.5 * (1.0 + math.cos(step * math.pi / self.adapt_steps))
            self.current_lr = self.min_lr + (self.current_lr - self.min_lr) * cosine_factor

        return self.current_lr
