# Calibration-domain crossover diagnostic

Fixed layer 3, retained payload 40%, no rank selection or recovery. Same evaluation tokens in both calibration conditions.

| Evaluation | Calibration | SVD delta NLL | Tucker r64 delta NLL | Tucker r48 delta NLL |
|---|---|---:|---:|---:|
| c4 | WikiText | +0.02625163 | +0.05161624 | +0.05205419 |
| c4 | C4 | +0.02548912 | +0.04999116 | +0.04906358 |
| wikitext | WikiText | +0.05235105 | +0.08061040 | +0.03832068 |
| wikitext | C4 | +0.05774123 | +0.08759704 | +0.05100194 |

c4: relative calibration effect {'mean': -0.0022280972064591437, 'paired_95_ci': [-0.0040624760951548135, -0.0002554873780961954]}; new Tucker48 minus SVD {'mean': 0.02357446610810936, 'paired_95_ci': [0.01892374554492254, 0.02900009106283317]}.

wikitext: relative calibration effect {'mean': 0.007291077167787194, 'paired_95_ci': [0.0005989083105336857, 0.014640481784619006]}; new Tucker48 minus SVD {'mean': -0.006739289793249663, 'paired_95_ci': [-0.014338006898614959, -0.00018769159708935276]}.

Diagnostic reuse of observed evaluations, one calibration sample per corpus. Corpus, sampling/packing and sample effects are not separated. Intervals conditional on factors and evaluated items; WikiText adjacent blocks may be correlated. Not a fresh confirmation test.

Baselines match previous runs exactly; ranks and payload unchanged; new factors identical across both evaluations. No inference memory/speed claim.
