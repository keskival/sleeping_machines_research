# Compact clock context for native guidance preparation

13 October 2026, B10. Compact native preparation now costs793.655MFLOPs,22.33%below the completed1021.878MFLOP reference; actor-only acquisition costs288.973MFLOPs. Preparation remains the dominant short-episode cost. The measured forecast gains and deployment activity savings do not justify charging unused controller training to standalone controls.

The early controller observes four own-clock gradient projections. On unaudited acquisition windows, the current source driver computes all own-parameter traces solely to obtain these summaries, then separately computes exact BPTT for the actor's admitted acquisition update. This is unnecessary derivative scope, not evidence against the native controller architecture.

For unit i in layer j, the factual complex state has form z_plus=A(alpha,omega,elapsed)*z_old+g(x)*W*x. Holding the incoming content fixed for the declared own-leg derivative, alpha=log_rate and omega affect only A. Their two state-sensitivity columns obey

    T_plus=A*T_old+[dA/dalpha*z_old, dA/domega*z_old].

If c is the current loss cotangent of the newly written state, the own-clock gradient is c^T*T_plus. Summing over units with1/sqrt(n) gives the exact four summaries consumed by the existing controller. This includes the current clock derivative and earlier own-state transport; it intentionally omits cross-leg historical derivatives, matching the declared own baseline. Exact actor acquisition separately differentiates its retained factual window, preserving the original full actor update.

The implementation carries4*n scalars per layer:2complex state coordinates times2clock parameters per unit. At two16-unit layers this is128trace floats versus7280for the full own-parameter bank. Current-window state cotangents are extracted from the retained factual graph; the actor's exact gradient is computed afterward. Arrays, cotangents and factual graph storage are additional; this is not a whole-model memory reduction or a replacement for deployment credit.

Three actual32-target acquisition updates reproduce exact actor gradients and forecasts with0error. The largest four-summary discrepancy is5.551e-17 against the full own implementation. Integrated2048-target acquisition uses compact context only on unaudited source windows and omits unused source remote banks:43to32. Acquired actor/Adam state matches exactly; controller parameters within8.882e-16. Every16384deployment score/request is unchanged. Executed Torch+NumPy preparation arithmetic is793.655MFLOPs including32768explicit frequency products, saving228.223MFLOPs; extra guidance cost beyond the common actor falls732.905to504.682MFLOPs. Audited windows retain both full own/projected gradients and actual Adam proposals. Special-function/Pythonmath/random internals remain separately unpriced. This is a preparation-stage arithmetic reduction, not full-lifetime/physical traffic/energy supremacy.
