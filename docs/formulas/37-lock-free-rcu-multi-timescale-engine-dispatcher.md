[Back to theory contents](../THEORY.md)

### Section 37: Lock-Free RCU Multi-Timescale Engine Dispatcher
Decoupling execution across temporal scales ($`\epsilon_0 \sim 10^{-3}\text{ s}`$, $`\epsilon_1 \sim 1\text{ s}`$, $`\epsilon_2 \sim 10^5\text{ s}`$) using Read-Copy-Update (RCU):

```math
\mathrm{Thread}_{\mathrm{fast}} \to \mathrm{Read}(\mathrm{ActivePtr})
```

```math
\mathrm{Thread}_{\mathrm{slow}} \to \mathrm{Write}(\mathrm{ShadowCopy}) \to \mathrm{AtomicSwap}()
```

Guarantees deadlock-free and livelock-free continuous inference.

---
