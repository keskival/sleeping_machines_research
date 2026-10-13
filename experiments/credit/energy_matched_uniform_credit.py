"""Native uniform residual with the addressed model's residual energy bound."""
from native_optimizer_receipts import ReceiptCredit
class EnergyMatchedUniformCredit(ReceiptCredit):
    def forward(self,features,observed,carry,operator):
        output,carry=super().forward(features,observed,carry,operator)
        return self.last_parent[None]+.25*(output-self.last_parent[None]),carry
