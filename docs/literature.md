# Public literature and implementation provenance

State slice: `aligned-holistic-continual-learning-clean-room-v1`.

These references define public methods or limitations that informed the
interfaces. They are reading material, not evidence imported into the
synthetic result. No source repository was copied into this clean-room tree.

| Topic | Public source | Use in this repository |
| --- | --- | --- |
| Activation caching and editing | [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | Motivates explicit activation/intervention boundaries; no code copied. |
| Sparse autoencoders | [SAELens](https://github.com/decoderesearch/SAELens) | Reference for feature dictionaries and reconstruction checks; not used by the synthetic runner. |
| Attribution graphs and interventions | [circuit-tracer](https://github.com/safety-research/circuit-tracer) | Informs the distinction between attribution, intervention, and causal validation. |
| Automated circuit discovery | [ACDC](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) and [paper](https://arxiv.org/abs/2304.14997) | Motivates conservative circuit claims and explicit control conditions. |
| Attribution patching | [Attribution Patching](https://arxiv.org/abs/2404.15255) | Public method context for cheap causal estimates; this project uses a simpler fixed synthetic intervention. |
| PyTorch tracing | [NNsight](https://github.com/ndif-team/nnsight) | Future model-backed tracing reference; model execution is closed here. |
| Intervention framework | [pyvene](https://github.com/stanfordnlp/pyvene) | Future interface reference for named interventions; no dependency here. |
| Gemma feature assets | [Gemma Scope](https://deepmind.google/discover/blog/gemma-scope-open-gemma-2-features-for-circuits-research/) | Asset provenance reference only; no model or feature asset is downloaded. |
| Continual-learning regularization | [EWC](https://arxiv.org/abs/1612.00796), [GEM](https://arxiv.org/abs/1706.08840), [LwF](https://arxiv.org/abs/1606.09282) | Conceptual baselines for future stream studies; no claims are made here. |
| Low-rank adaptation | [LoRA](https://arxiv.org/abs/2106.09685) | Future adapter-control reference; no adapter training is implemented. |
| Interpretability limitations | [Anthropic attribution-graph methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Records known gaps: reconstruction dark matter, missing circuits, feature abstraction, and compounding intervention error. |
| Public feature browser | [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) | Optional future visualization reference; not an evidence store. |

Before a model-backed continuation, each selected source must be pinned to an
exact revision, license recorded, and its role restricted to a public method
or a reproducible asset. A citation never substitutes for an independent
review or held-out evidence.
