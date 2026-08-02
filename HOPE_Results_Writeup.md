# Compression as Cartography: HOPE, Tested Against Language

Results from a verified NumPy implementation of Mobahi
and Bartlett's "Hilbert Operator for Progressive Encoding" (arXiv:2607.21366),
put on trial against a character-level language model, three corpora of
increasing breadth, and GPT-2.

When I first read HOPE, I closed my response with a question the paper left
open: whether the Gaussian surrogate survives the Zipfian skew of text
activations. This is the answer, or at least the beginning of one. The short
version: the surrogate dies in correlation everywhere, survives in rank
wherever its shape assumption holds, and in the one place that assumption
collapses outright, the rank itself inverts. The ordering it assigns to
neurons is durable across corpora and most of GPT-2's depth; the redundancy
it perceives between neurons never once changes a decision; and in GPT-2's
most heavily non-Gaussian block, the data-free map is not merely noisy but
anti-guidance. The map, drawn anyway, turns out to be
real, robust, and legible; and its legibility is register-relative in
exactly the way the paper's transfer method presupposes.

## The instrument

The claim under test is that a neuron's identity is the function it computes
over a distribution of inputs, not the parameters that happen to encode it.
Formally, neuron $i$ with input weights $w_i$, bias, and output weights
$w^{out}_i$ is lifted to a rank-1 operator whose norm, its **capacity**, is

$$\lVert f_i \rVert \;=\; \lVert w^{out}_i \rVert \sqrt{K_{ii}}, \qquad
K_{ii} \;=\; \mathbb{E}\!\left[\phi(y_i)^2\right],\; y_i \sim \mathcal{N}(\beta_i, \gamma_i^2).$$

Everything downstream depends on evaluating such kernels without data. The
surrogate declares each pre-activation Gaussian with calibration-measured
mean and standard deviation; for ReLU the self-kernel is closed-form
(the paper's Eq. 3),

$$K_{ii} \;=\; (\beta^2+\gamma^2)\,\Phi(\beta/\gamma) \;+\; \beta\gamma\,\varphi(\beta/\gamma),$$

pairwise correlation comes from weight geometry through the Maximum-Entropy
warp (Eq. 4), solving $\hat\rho/(1-\hat\rho^2)=\kappa$ with

$$\kappa \;=\; \frac{\rho_{\mathrm{eff}}}{1-\rho_{\mathrm{eff}}^2}\, r_i r_j,
\qquad r_k = \sigma_k / \lVert w_k \rVert,
\qquad \hat\rho = \frac{2\kappa}{1+\sqrt{1+4\kappa^2}},$$

and the cross-kernel follows the arc-cosine form (Eq. 5),
$K_{ij} \approx J_1(\hat\rho)\sqrt{K_{ii}K_{jj}}$ with
$J_1(\rho) = \left(\sqrt{1-\rho^2} + (\pi - \arccos\rho)\,\rho\right)/\pi$.
Pruning and merging are then priced in one currency (Eq. 6): with layer
capacity $E_a$,

$$J_{\mathrm{prune}}(i) = \frac{N\lVert f_i\rVert}{E_a - \lVert f_i\rVert},
\qquad
J_{\mathrm{merge}}(i,j) = \frac{N \cdot D}{E_{\mathrm{rem}} + s^{\ast}},$$

where the optimal parent magnitude and distortion are
$s^{\ast} = (a + b\,E_{\mathrm{rem}})/(2E_{\mathrm{rem}} + b)$ and
$D^2 = a + 2{s^{\ast}}^2 - 2 s^{\ast} b$, with $a$ the pair's Hilbert energy, $b$ its
alignment with the synthesized parent direction (the principal right-singular
vector of $w^{out}_i \tilde w_i^{\top} + w^{out}_j \tilde w_j^{\top}$, computed
by the rank-2 reduction), and $E_{\mathrm{rem}}$ the capacity of everything
else. Every closed form in the implementation is pinned by Monte-Carlo tests;
for GPT-2, whose GELU sits outside the PH-1 family the closed forms cover,
the same kernels are evaluated by Gauss-Hermite quadrature under the same
surrogate, pinned both against Monte Carlo and against the ReLU closed forms.

The experimental design was staged: build and verify the engine; put the
surrogate on trial (are pre-activations Gaussian? do the kernels track
reality?); then draw the map, deploying every action into the live model so
the ledger doubles as a loss trajectory, with each unit carrying a stable
identity, a linguistic label, and its measured kurtosis. Each mapping run
was performed twice, once with the paper's warped correlations and once
with correlations measured on the calibration slice, to isolate how much of
the map depends on the correlation model. The corpora: tiny Shakespeare
(1.1M characters), the MASC slice of the Open American National Corpus
(2.9M characters, 20 genres), and the full OANC (88.5M characters, 8
genres), with GPT-2 small as the transformer frontier.

## Finding 1: the fiction holds in the middle and fails in the tail

On every model tested, the median unit is close to Gaussian and the tail is
not. The char-MLP's median excess kurtosis is $-0.008$ (max $+3.1$); MASC
gives $-0.030$ (max $+7.3$); full OANC $+0.008$ (max $+7.3$). GPT-2 is the
same story amplified: pooled median $+0.43$, maximum $+776.6$, with the
non-Gaussianity concentrated at the ends of the stack (block 2's median
excess kurtosis is $+2.4$; the mid-stack plateaus near $+0.22$; block 12
rises again). The Zipfian skew of text does reach the activations, but it
arrives as a heavy tail of outlier units, not as a failure of the typical
case. Kernel values feel it: self-kernel errors run 7 to 17 percent across
the corpora and cross-kernels 9 to 25 percent, growing with breadth.

## Finding 2: distances distort; the ordering survives

The number the framework actually spends is not the kernel value but the
ranking it induces. Capacity orderings correlate with empirically measured
capacities at Spearman $0.982$ (Shakespeare), $0.963$ (MASC), and $0.938$
(OANC): a visible decay with distributional breadth, entangled with the
models being better trained, but an ordering that remains firmly intact. A
stranger regularity replicated three times: the full data-free pipeline
predicts cross-kernels *better* than the same pipeline given oracle
correlations, because the MaxEnt warp's shrinkage partially cancels the
zero-bias approximation's overshoot. Two fictions, leaning on each other.

## Finding 3: the warp silences consolidation

The headline. Under the paper's correlation model the encoder **never
merges**. Across all three corpora, every single action chosen was a prune:
167 of 167, 250 of 250, 249 of 249. The mechanism is visible in one pair of
numbers: on the char-MLP the warp caps pairwise correlation at $0.13$ where
the empirical maximum is $0.67$. No pair ever looks redundant enough for a
merge to outprice a prune. Replace $\hat\rho$ with measured correlations,
leaving every other step untouched, and merging wakes: 17, 41, and 76 merges
respectively, with slightly better loss trajectories. The framework's
elegant unification of pruning and merging into one projection is, on
text-trained models under its own surrogate, a unification in principle
only; removal is priced truthfully, however redundancy is not perceived.

## Finding 4: the map is real, robust, and corpus-relative

Judged as compression, the capacity ordering earns its keep where there is
slack to find: HOPE's trajectory sits $0.47$ to $0.55$ nats below L1-norm
pruning at 35 percent density on the smaller corpora, and $0.43$ below on
full OANC. At full scale a caveat appears that I consider a finding: random
pruning ties capacity-ordered pruning ($2.638$ vs $2.636$ nats), because a
saturated model has no cheap units, so every order costs alike. The value of
the ordering then concentrates entirely in what survives rather than the
curve, which is the cartographic point anyway.

Judged as measurement, the map barely moves when the correlation model is
swapped: removed-set Jaccard $0.898$, $0.912$, $0.789$ and removal-order
Spearman $0.974$, $0.946$, $0.899$ across the three corpora. And what the
map says is legible and corpus-relative in the way a map should be. On
Shakespeare, structure survives: the newline and speaker-header detectors
(one unit's top-activating context is literally `\nWARWICK`) are among the
best preserved, while punctuation detectors are cut hardest. On OANC the
same architecture, allocated against different territory, evicts its newline
units entirely; the corpus defines what counts as load-bearing. Heavy-tailed
units skew toward surviving in every run, consistent with outlier features
being high-capacity detectors of frequent structure: the transformer
outlier-feature worry attaches to mispricing, not to being pruned first.

## Finding 5: the register-conditional map, or DEFT's premise measured

Because the surrogate is defined by whatever distribution calibration sees,
the same trained weights can be mapped per register. On OANC's genres this
yields a mostly shared core (median cross-register capacity Spearman
$0.979$, top-half core overlap $0.90$) with a measurable, legible fringe:
the poles are conversation and technical prose (minimum Spearman $0.855$,
core overlap $0.745$), and the most register-variable units read exactly as
they should, transcript-formatting detectors core in conversation, biomedical
text detectors core in technical writing, an all-caps onset detector core in
movie scripts and slack in travel guides. Which knowledge is core is,
measurably, a property of the register the surrogate protects. That is the
premise DEFT's frozen-core transfer stands on, demonstrated on real
registers with nameable units.

## Finding 6: GPT-2, where the fiction's failure reaches the map

The GPT-2 experiment (blocks 2, 6, and 12 encoded to 35 percent density,
OANC calibration and evaluation, quadrature GELU kernels) was designed
around one hypothesis: paper-versus-empirical map divergence should track
the depth profile of non-Gaussianity. The hypothesis resolved through a
different channel than posed, twice over.

First, correlations never mattered. Zero merges under both correlation
models in every block; removed-set Jaccard 1.000 three times; the paper,
empirical, and capacity-prune-only trajectories are pixel-identical. The
mechanism is width: a 3072-unit block carries so deep a reservoir of
near-dead units that the cheapest prune undercuts every candidate merge's
distortion all the way down to 35 percent density. On the char-MLP the warp
blinded the encoder to redundancy; on GPT-2, consolidation is outcompeted
even with redundancy in plain view. The correlation half of the framework is
moot at transformer width, at least at these densities.

Second, the fiction's failure surfaced in the capacity pricing itself, and
it surfaced exactly where the shape statistics said to look. In block 2,
whose median excess kurtosis is +3.09 under OANC calibration, the data-free
ordering is anti-guidance: following it drives NLL from 3.56 to roughly 5.15
by 60 percent density, then recovers non-monotonically to 4.3, while
random deletion of the same fractions costs almost nothing. Two things in
that sentence deserve emphasis. Random being free means block 2 is highly
redundant to unbiased deletion, so the surrogate is not failing to find
slack; it is systematically selecting load-bearing units as slack. And the
non-monotone recovery means the damage involves interactions between units
that no per-unit pricing can represent. Block 6's ordering trails random
mildly; block 12's beats every baseline. Three blocks are three data points,
so the honest claim is locational rather than dose-response: the
catastrophic inversion co-locates with the extreme non-Gaussianity, and
where the Gaussian fiction roughly holds, the ordering is competitive to
winning.

The register-conditional map, meanwhile, gained a depth axis, and it is the
run's most consequential result: cross-register capacity Spearman declines
near-monotonically from about 0.90 in blocks 1 through 3 to 0.48-0.55 in
blocks 10 through 12 (with a small end-of-stack uptick). Register-universal
foundations; register-specific depths. The profile is decoupled from the
kurtosis curve, which spikes early while register-specificity accumulates
late, so the depth gradient is not a heavy-tail artifact. For DEFT this
turns "freeze a core" into a measured, depth-resolved partition: for
register adaptation, the deep feed-forward blocks are what you would open.
For anyone calibrating a data-free map, it yields a rule of practice: the
choice of calibration corpus barely matters early in the stack and matters
enormously by block 10.

## What it means, and what it doesn't

For HOPE: the correlation model fails conservatively (missing merges rather
than making bad ones) and, at transformer width, fails into irrelevance;
the capacity model is the load-bearing joint, trustworthy across corpora and
across most of GPT-2's depth, and anti-trustworthy precisely where the shape
assumption collapses. The practical diagnostic is already in hand: the same
calibration pass that builds the surrogate yields the per-block kurtosis
profile that flags where not to believe it. For language models: compression order is readable as a hierarchy
of dispensability, that hierarchy is corpus- and register-relative, and the
eviction ledger gives the "acquisition in reverse" reading an operational
form. The limits being these are small models mapped without retraining
between actions (deliberately, to read the raw data-free trajectory);
labels are cheap correlational probes; the breadth trend is confounded with
training quality; and the GELU parent synthesis fixes an input scale that
ReLU's homogeneity would have left free.

The paper's coda calls parameters mere shadows of the function. Held up
against language, the shadows cast a map, and the map reads, provided you
check the projection before you navigate: one calibration pass's kurtosis
profile tells you where the chart is sound and where it is drawn upside
down. 
