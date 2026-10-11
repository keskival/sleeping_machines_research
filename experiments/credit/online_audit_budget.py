"""Selected-route audit budget dual: fixed-size state, no candidate enumeration.
Average-work controller, not a hard per-event memory/work limit.
"""
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class BudgetPacket:
    propensity: float
    cost: float
    version: int
    log_price: float

class AuditBudget:
    def __init__(self,budget,*,log_price=0.,floor=.02):
        if not math.isfinite(budget) or budget<=0 or not 0<floor<=1 or not math.isfinite(log_price):raise ValueError('invalid budget')
        self.budget=budget;self.log_price=log_price;self.floor=floor;self.version=0
    def issue(self,risk,cost):
        if not math.isfinite(risk) or risk<0 or not math.isfinite(cost) or cost<=0:raise ValueError('invalid selected-route risk/cost')
        lograte=math.log(self.floor) if risk==0 else .5*(math.log(risk)-math.log(cost)-self.log_price)
        rate=math.exp(min(0.,max(math.log(self.floor),lograte)))
        return BudgetPacket(rate,cost,self.version,self.log_price)
    def consume(self,packet,*,audited,step=.01):
        if not math.isfinite(step) or step<=0:raise ValueError('invalid step')
        actual_work=packet.cost if audited else 0.
        # Positive overspend raises audit price; stale packet is a past observation.
        updated=self.log_price+step*(actual_work-self.budget)
        if not math.isfinite(updated):raise ValueError('nonfinite budget update')
        self.log_price=updated;self.version+=1
        return actual_work-self.budget
