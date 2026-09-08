# Public literature and implementation provenance

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

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
| Gemma feature assets | [Gemma Scope](https://deepmind.google/discover/blog/gemma-scope-open-gemma-2-features-for-circuits-research/) | Historical asset-provenance reference; the integrated slice uses a separately pinned GPT-2 SAE. |
| Continual-learning regularization | [EWC](https://arxiv.org/abs/1612.00796), [GEM](https://arxiv.org/abs/1706.08840), [LwF](https://arxiv.org/abs/1606.09282) | Conceptual baselines for future stream studies; no claims are made here. |
| Low-rank adaptation | [LoRA](https://arxiv.org/abs/2106.09685) | Future adapter-control reference; no adapter training is implemented. |
| Interpretability limitations | [Anthropic attribution-graph methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Records known gaps: reconstruction dark matter, missing circuits, feature abstraction, and compounding intervention error. |
| Public feature browser | [Neuronpedia](https://github.com/hijohnnylin/neuronpedia) | Optional future visualization reference; not an evidence store. |

Before a model-backed continuation, each selected source must be pinned to an
exact revision, license recorded, and its role restricted to a public method
or a reproducible asset. A citation never substitutes for an independent
review or held-out evidence.

## Integrated source pins

The source forks are Git submodules, not copied files. Their exact commit
identities and license files are machine-readable in
`third_party/UPSTREAMS.lock.json`:

| Source | Pinned commit | License identity |
| --- | --- | --- |
| TransformerLens | `2a05c15df238caf8b7869188ea27b5d39a903d35` | MIT, `LICENSE` |
| TransformerLens circuit-tracer legacy | `a634e5757cd2ae43add9beb492be7a2dfb443a30` | MIT, `LICENSE` |
| TransformerLens ACDC legacy | `3745d0c18b683cdcdd4b4acf3800f038c2fe8aa7` | MIT, `LICENSE` |
| SAELens | `ee45e7406165ce267b33d33b7371bbb63ef24db8` | MIT, `LICENSE` |
| circuit-tracer | `6018ed8d35e40f2c50062822e8dde422b8e52e2d` | MIT, `LICENSE` |
| NNsight | `87f4dadaae254c6d36e8820cc83a251623b8dda7` | MIT, `LICENSE` |
| NNsight circuit-tracer legacy | `3fd13f8959cf9bcb79fc87d24bc244bceb26e3e5` | MIT, `LICENSE` |
| pyvene | `9e333904dcf9e597ca76170010d17f4d4580de8d` | Apache-2.0, `LICENSE` |
| ACDC | `bc99ace817974b5584b7ee203d596a8e2bbcd399` | MIT, `LICENSE` |
| Neuronpedia | `df03cfee33e9023866eba6f828d9e34900932877` | Apache-2.0 plus `NOTICE` and `LICENSE-MIT` |

The pinned sources have incompatible dependency lanes: TransformerLens at this
revision requires Transformers 5.9 or newer, circuit-tracer requires 4.57.3,
and ACDC requires Transformers 4.35 with Torch below 2.0. The integrated design
isolates the modern, circuit-tracer, and ACDC environments rather than claiming
a single installable environment.
