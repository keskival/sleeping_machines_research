"""Native future-directed adjustment alongside complete exact actor credit."""
import torch
from native_market_credit import NativeMarketCredit
class NativeProspectiveMarketCredit(NativeMarketCredit):
    def __init__(self,width):
        super().__init__(width)
        with torch.random.fork_rng():
            head=torch.nn.Linear(self.core.total_payload,width//3)
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
        self.core.head=head;self.core.classes=width//3
    def forward(self,start,events,carried):
        output,carried=super().forward(start,events,carried)
        return torch.cat((torch.zeros_like(output),output),-1),carried
