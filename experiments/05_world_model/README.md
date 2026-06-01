# Tiny World Model Experiments

This folder is reserved for experiments using the lightweight world-state model.

The reusable module lives in:

```text
src/world_model/
```

Current intended interface:

```text
current world state + candidate robot action
    -> predicted next world state
    -> safe/risky assessment
```

For now, this module should stay independent from the main OpenVLA code:

- no OpenVLA imports
- no model checkpoint dependency
- no GPU requirement
- no Hugging Face or transformers dependency
- no direct robot policy dependency
