"""Joint native prospective credit and optimizer control, one shared event core."""
import torch
from native_market_credit import NativeMarketCredit
class NativeOptimizerMarketCredit(NativeMarketCredit):
    def __init__(self,width):
        super().__init__(width)
        with torch.random.fork_rng():
            head=torch.nn.Linear(self.core.total_payload,width//3+1)
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        self.core.head=head;self.core.classes=width//3+1
    def forward(self,start,events,carried):
        output,carried=super().forward(start,events,carried)
        return torch.cat((torch.zeros_like(output[:,:-1]),output),-1),carried
