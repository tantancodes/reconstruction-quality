# How the set models work, layer by layer

Code: `geo_models.py` (models), `run_experiments.py` (training and evaluation). Shapes use
B = batch, N = cameras (padded to the largest set, with a boolean mask), h = hidden width,
H = heads (4), d = h / H, R = pairwise relation features.

## 1. What goes in: camera geometry

Metashape gives each aligned camera a centre `c_i` (3 numbers) and a viewing direction `v_i`
(unit vector, 3 numbers). A build without markers comes out in an arbitrary frame: unknown
rotation, translation and scale. Translation and scale are removed per build
(`pilot_set_models.load_runs`):

    p_i = (c_i - mean(c)) / median_k |c_k - mean(c)|

So the input is `X ∈ R^{B×N×6}` (real data: position + view) or `R^{B×N×3}` (synthetic:
position only), with mask `M ∈ {0,1}^{B×N}`. Rotation is *not* removed, because there is no
canonical orientation without a reference. The frame-dependent models learn to ignore it from
augmentation: every training example is multiplied by a fresh random rotation `Q ∈ SO(3)` each
step (`rotate()`), so the only thing that is stable across steps is the layout itself.

The full camera *pose* is a rigid transform in SE(3): rotation plus translation. We use the
centre and the optical axis only (5 degrees of freedom per camera, roll about the axis is
dropped), which is what matters for coverage.

## 2. Permutation invariance

A capture has no natural camera order, so the prediction must not change if the cameras are
listed in a different order. All models have the form

    y = rho( POOL_i  phi(x_i, context) )

where POOL is an elementwise max over cameras (`.max(1)`). Max over a set does not depend on
order, so the output is permutation-invariant. Padded cameras are filled with -1e9 before
the max so they can never win.

## 3. PointNet (`PointNetSet`)

    phi:  (B,N,in) -> Linear -> ReLU -> Linear -> ReLU -> (B,N,h)
    pool: max over N                                       -> (B,h)
    rho:  Linear -> ReLU -> Linear                         -> (B,1)

Each camera is encoded on its own. The pooled vector can say "some camera is high", "some
camera is far left", but it cannot directly represent "the gap between neighbouring cameras",
because no camera ever sees another before pooling. That is why PointNet struggles on the
localized task.

## 4. Self-attention (`AttnSet` with rel_dim = 0, identical to the existing model)

Embed: `f = embed(x)`, (B,N,h).

Queries, keys, values are three linear maps of the same tokens (`self.qkv`, one Linear that
outputs 3h, then reshaped):

    Q = f W_Q,  K = f W_K,  V = f W_V,   each (B,N,h), split into H heads -> (B,H,N,d)

Scaled dot-product attention, per head:

    logits = Q K^T / sqrt(d)                (B,H,N,N)   how much camera i should look at camera j
    W      = softmax over j (masked)        (B,H,N,N)   each row sums to 1
    out    = W V                            (B,H,N,d)   weighted average of the other cameras' values

Dividing by sqrt(d) keeps the dot products at unit scale when d grows; without it the softmax
saturates and gradients vanish. Masked keys get -inf, so padded cameras receive zero weight.

Multi-head: the H heads run in parallel on d = h/H dimensions each, then are concatenated back
to (B,N,h) and mixed by `self.out`. Different heads can learn different relations (e.g. one
attends to neighbours in azimuth, another to cameras on the other ring).

Then a residual `f + attn(f)`, a feed-forward block (`ff`), max-pool and `rho`, as in PointNet.

`test_geo_models.py` checks that `GeoAttention` with no bias gives the same output as
`torch.nn.MultiheadAttention` with the same weights. The two differ in initialisation only: `nn.MultiheadAttention` starts
its input projections Xavier-uniform with zero biases, `GeoAttention` uses the default `nn.Linear`
initialisation. That changes training outcomes (13.1° vs 16.1° on the same seed at 400 epochs; see
REPORT.md finding 2).

## 5. Geometry-aware attention (`geo`)

Only one line changes (`GeoAttention.forward`):

    logits[b,k,i,j] = Q_i · K_j / sqrt(d) + g_k(r_ij)

`r_ij ∈ R^R` are pairwise relations computed once from the normalised positions
(`relations()`):

| feature | formula | meaning |
|---|---|---|
| d_ij | ‖p_i − p_j‖ | distance between the two cameras |
| c_ij | r̂_i · r̂_j, r̂ = p/‖p‖ | cosine of their angular separation seen from the object |
| u_ij | v_i · v_j (real data only) | do they look the same way |

`g` is a 2-layer MLP (R → 16 → H) shared over all pairs, giving one bias per head. It adds
about 100 parameters (R=2: 2·16+16+16·4+4 = 116) to a model of ~22k, so it is not a capacity
increase. This is a *relative* encoding: it says how two cameras relate, not where either one
is, in the same spirit as relative position bias in Swin Transformer and the spatial encoding in
Graphormer.

Why it should help: the targets depend on relations. The largest coverage gap is the largest
angular separation between azimuthally neighbouring cameras, i.e. a function of the c_ij. Plain
attention has to reconstruct c_ij from the dot product of two learned embeddings of raw,
randomly rotated coordinates, and has to learn rotation invariance from augmentation. With the
bias, the invariant relation enters the logits directly, so a head can learn "attend to my
angular neighbours" with a few parameters.

Mathematically, d_ij, c_ij and u_ij are unchanged by any rotation Q (‖Qa − Qb‖ = ‖a − b‖, and
Qa · Qb = a · b), so the bias term is exactly rotation-invariant (`test_relations_are_rotation_invariant`).

## 6. The ablations

| model | tokens | pairwise bias | what it tests |
|---|---|---|---|
| attention | raw coordinates | none | the existing model |
| geo | raw coordinates | true relations | does relative geometry help |
| geo_shuf | raw coordinates | relations shuffled within the set | same parameters and compute as geo; if geo only wins because of the extra MLP or extra compute, geo_shuf should win too |
| geo_inv | only ‖p_i‖ (and v_i·r̂_i) | true relations | is the whole model better when it is rotation-invariant by construction instead of by augmentation |
| xgb_pair | — | the same angular relations as a histogram + nearest-neighbour stats | can pairwise geometry *without* attention do as well |

## 7. Loss, backpropagation, optimisation

Target: the label divided by a constant (synthetic: degrees / 100; real: label / training-fold
mean), so the network outputs numbers near 1. Loss: mean squared error. Backpropagation
computes the gradient of the loss with respect to every weight by the chain rule through
rho, the max (gradient flows only to the camera that won each channel), the ff block, the
softmax, and Q, K, V. Adam updates each weight with a step scaled by running estimates of the
gradient's mean and variance. Synthetic runs: lr 3e-3 with cosine decay to 0 over 150 epochs,
batch 128 (or lr 1e-3, 400 epochs, as in the original benchmark, for the long check). Real
runs: lr 1e-3, weight decay 1e-4, full batch, 400 epochs, exactly as the pilot.

Reported numbers use the weights at the last epoch (no early stopping on the test set). The
validation set only records curves and picks tabular hyperparameters.

## 8. Leakage control

Synthetic: train, validation and test sets are independent draws; count_shift tests on camera
counts never seen in training. Real: builds from one capture share photos, so leave-one-run-out
can leak (a near-duplicate build is in training). The main real splits hold out a whole capture
group (dog old pipeline / dog new pipeline / coffin) or a whole object.
