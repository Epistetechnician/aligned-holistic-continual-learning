# Document-level power design

State slice: `aligned-holistic-continual-learning-integrated-stack-v2`.

`ahcl.power_design.PowerDesign` plans a paired contrast where the independent
sampling unit is a document. A document may have repeated endpoint
measurements, but repeats do not create independent observations. The
standardized effect `d` is defined at the document paired-contrast level, so
the normal approximation uses noncentrality `d * sqrt(N)`, where `N` is the
number of independent documents. Repeats are validated metadata and do not
alter variance; applying an ICC design effect here would double count
within-document aggregation.
For `K` endpoints, the planner uses the conservative per-contrast level
`alpha / K`. This is conservative for a later Holm procedure; it does not
claim that Holm and Bonferroni have identical power.

`marginal_required_clusters` answers whether each contrast individually reaches
the target power. If `joint_target` is supplied,
`joint_required_clusters` answers a separate question: whether the
Bonferroni union-bound lower bound reaches that joint target. The bound is
`max(0, 1 - K * (1 - marginal_power))` and makes no independence assumption
among endpoints.

All inputs are finite; counts (`endpoints`, `repeats`, and the bounded search
limit) are positive integers. The search fails closed when the requested count
exceeds `max_clusters`.

This planner is a design calculation. It does not establish that any current
six-document panel or fit/tune/assessment split has sufficient power. For the
current four-axis, three-repeat design with a standardized effect of 0.50, the
conservative joint lower bound requires more than six independent documents.
The planner does not authorize model execution or scientific assessment.

For an illustrative design with standardized document effect `0.5`, four
endpoints, and three recorded repeats, the planner yields per-contrast
`alpha = 0.0125`, 45 documents for 0.80 marginal power, and 69 documents for
the 0.80 union-bound joint target. Six documents produce marginal power about
`0.102` and a zero-clipped joint lower bound of `0`; these are planning values,
not a scientific adequacy claim.
