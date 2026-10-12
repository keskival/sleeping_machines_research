"""Predictive-law side information enters the same native addressed event core."""
import torch
from native_optimizer_market_credit import NativeOptimizerMarketCredit
class NativeForecastMarketCredit(NativeOptimizerMarketCredit):
    forecast_width=162
    def __init__(self,width):
        super().__init__(width)
        with torch.random.fork_rng():
            self.forecast_adapter=torch.nn.Linear(self.forecast_width,self.core.total_payload,bias=False)
            torch.nn.init.zeros_(self.forecast_adapter.weight)
    def forward(self,start,events,carried):
        assert start.shape[-1]==self.width+self.forecast_width
        # The causal boundary response is constant throughout this observed group.
        extra=self.forecast_adapter(start[0,self.width:])
        hook=self.core.content.register_forward_hook(lambda module,args,out:out+extra)
        try:return super().forward(start[:,:self.width],torch.cat((events[:,:,:self.width],events[:,:,-8:]),-1),carried)
        finally:hook.remove()
