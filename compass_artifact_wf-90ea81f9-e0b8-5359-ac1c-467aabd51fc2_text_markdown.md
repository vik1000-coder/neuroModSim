# Graph-Based and Dynamics-Based Models of Neural Dynamics, with Emphasis on Neuromodulation and State-Dependent Effective Connectivity

## TL;DR
- The field is organized around one master picture: a fast neural flow **dx/dt = F(x; W, m)** coupled to slow modulator dynamics **dm/dt = εG(x, m)**. Three distinct "graphs" must be kept separate: the **structural graph W** (anatomy, multilayer), the **effective graph J(x,m) = D_xF** (state-dependent Jacobian — your estimand), and the **functional graph** (statistical dependence). Most classical connectome models fix W and ignore m; the 2023–2026 frontier is precisely about how m reshapes J.
- Neuromodulation enters models at four mathematically distinct points — (a) gain rescaling (diagonal, multiplicative), (b) weight/multilayer reconfiguration (additive extra edges), (c) intrinsic-property/bifurcation change, and (d) full two-timescale coupling — and these map directly onto the D(σ)J₀ vs J₀+J₁ vs unstructured hypotheses you want to test.
- On the "kill search": learned dynamical/generative models that recover Jacobian-like effective connectivity from worm whole-brain data now exist (decomposed-LDS, published Comm. Biol. 2025; connectome-constrained LDS fit to Randi's data; a 2026 score/diffusion preprint), and Randi et al. 2023 is already used as a fitting/validation target. However, **no located work formally tests whether behavioral-state changes act as a diagonal gain D(σ)J₀ vs an additive layer J₀+J₁ vs unstructured change** — that specific comparative, modulator-indexed test appears open.

## Key Findings
1. **Fixed-W conductance models are mature but structurally blind to modulation.** The Kunert–Shlizerman–Kutz (2014) single-compartment conductance model on the White/Varshney connectome, its multistability/robustness descendants, and BAAIWorm (2024) all hold W fixed and reproduce low-dimensional attractor/limit-cycle behavior; none carry a modulator state m.
2. **Anatomy under-predicts function on seconds timescales.** Randi et al. (2023) show extrasynaptic (dense-core-vesicle/peptidergic) signaling drives fast transients absent from the wiring diagram, and their measured signal-propagation atlas predicts spontaneous dynamics better than anatomy — the empirical core of the "W is not J" argument.
3. **The extrasynaptic connectome is now mapped** (Bentley 2016 multilayer; Ripoll-Sánchez 2023 neuropeptidergic; Beets 2023 GPCR deorphanization), giving explicit additional layers W_c for the J₀+J₁ hypothesis.
4. **Neuromodulation has a clean mathematical taxonomy**: gain (Chance–Abbott–Reyes 2002; Shine 2018; Stroud 2018; Li 2019), weight reconfiguration (Marder; Bargmann–Marder 2013), intrinsic/bifurcation, and coupled neuronal–neurotransmitter systems (Kringelbach 2020).
5. **Novelty is partially anticipated but a well-defined gap remains** around modulator-indexed, structurally-decomposed effective-connectivity estimation validated against Randi.

## Details

### 0. Framing: the master equation and three graphs

Write the state of a nervous system as a fast variable x ∈ ℝⁿ (membrane potentials, calcium proxies, or rates of n neurons) and a slow variable m ∈ ℝᵏ (monoamine/neuropeptide concentrations or receptor-activation fractions). The general two-timescale system is

  dx/dt = F(x; W, m),  dm/dt = ε G(x, m),  0 < ε ≪ 1.  (0.1)

Here W is a (multilayer) wiring tensor and ε encodes timescale separation between fast electrical/synaptic dynamics and slower modulatory dynamics. Three graphs must be distinguished:

- **Structural graph W** — anatomy. In C. elegans this is itself multilayer: chemical synapses (directed, signed), gap junctions (symmetric, electrical), and extrasynaptic monoamine/peptide "wireless" layers. Sources: White 1986; Varshney 2011; Cook 2019; Bentley 2016; Ripoll-Sánchez 2023.
- **Effective graph J(x,m) = D_x F** — the Jacobian of the flow evaluated at a point (x,m). This is the state-dependent, modulator-dependent "who-drives-whom" matrix and is the reader's estimand: a point-indexed lagged effect of the neural flow. J determines local stability, growth directions, and lagged influence; it is generically NOT equal to W.
- **Functional graph** — statistical dependence in activity (correlation, covariance, transfer entropy, Granger). It is a downstream consequence of J, noise, and inputs, and can be non-zero between neurons with J_ij = 0.

The reader's program — score-based/generative inference of a point-indexed Jacobian on a history manifold — is precisely an attempt to estimate J(x,m) directly from activity, rather than assuming J = W (connectome models) or reading J off correlations (functional models).

### 1. Fixed-graph dynamical models

**1.1 Biophysical / conductance connectome models.**
Kunert, Shlizerman & Kutz (2014, Phys. Rev. E 89:052805) posed the first full-connectome single-compartment conductance model of C. elegans. Each neuron i obeys a current-balance equation

  C dVᵢ/dt = −G_c(Vᵢ − E_cell) − Σⱼ G^gap_ij (Vᵢ − Vⱼ) − Σⱼ G^syn_ij sⱼ (Vᵢ − Eⱼ) + Iᵢ^ext,  (1.1)

with synaptic activation

  dsⱼ/dt = a r(Vⱼ)(1 − sⱼ) − b sⱼ,  r(Vⱼ) = 1/(1 + exp(−κ(Vⱼ − V_th))).  (1.2)

Symbols: C membrane capacitance; G_c leak conductance, E_cell leak reversal; G^gap_ij gap-junction conductance (symmetric, from the electrical connectome); G^syn_ij chemical synaptic conductance (directed, from the chemical connectome); sⱼ ∈ [0,1] postsynaptic activation of neuron j; Eⱼ synaptic reversal potential (sets sign/excitatory–inhibitory); a,b rise/decay rates; r sigmoidal presynaptic activation; Iᵢ^ext injected/sensory current. W here is (G^gap, G^syn); m is absent. The linearization around a fixed point gives an effective graph J that mixes gap and synaptic terms and their signs. Descendants: Kunert-Graf, Shlizerman, Walker & Kutz (2017, Front. Comput. Neurosci. 11:53) on multistability and long-timescale transients; Kunert, Maia & Kutz (2017, PLoS Comput. Biol. 13(1):e1005261) on injured-connectome robustness; Kunert, Proctor, Brunton & Kutz (2017, PLoS Comput. Biol. 13(1):e1005303) on spatiotemporal feedback and locomotion. BAAIWorm (Zhao et al. 2024, Nat. Comput. Sci. 4:978–990) is a closed-loop brain–body–environment model with multicompartmental, ion-channel-resolved neurons (136 neurons in the foraging circuit) validated against patch-clamp for representative neurons. Nicoletti et al. (2024, PLoS ONE 19(3):e0298105, "Biophysical modeling of the whole-cell dynamics of C. elegans motor and interneurons families") provide biophysical whole-cell models of C. elegans motor- and inter-neuron families. The OpenWorm c302 framework (Gleeson et al.) provides configurable multiscale NeuroML models. *[Citation-status note: c302 primary reference not independently verified — see status section.]*

**1.2 Rate / neural-mass / whole-brain connectome models.**
Wilson–Cowan (1972) coupled excitatory/inhibitory populations:

  τ_E dE/dt = −E + f(w_EE E − w_EI I + P),  τ_I dI/dt = −I + f(w_IE E − w_II I + Q),  (1.3)

with f a sigmoid, w_** population couplings, P,Q external drives. Breakspear (2017, Nat. Neurosci. 20:340–352) reviews the passage from such local mass models to whole-brain models. The Virtual Brain platform simulates coupled neural masses on empirical structural connectomes. The Deco/Kringelbach Hopf ("Stuart–Landau") whole-brain model places a normal-form oscillator at each node:

  dz_j/dt = (a_j + iω_j) z_j − z_j |z_j|² + G Σ_k C_jk (z_k − z_j) + β η_j(t),  (1.4)

z_j ∈ ℂ the complex state of region j, a_j the bifurcation parameter (a_j<0 stable focus, a_j>0 limit cycle; Hopf at a_j=0), ω_j intrinsic frequency, C_jk structural connectivity, G global coupling, η noise. Dynamic mean-field models reduce spiking networks to regional firing rates driven by C_jk scaled by G.

**1.3 Linear / spectral, topology-only models.**
Network control theory (Yan et al. 2017, Nature 550:519–523) takes ẋ = Ax + Bu with A the (weighted) connectome, B input matrix, and uses the controllability Gramian to predict which neurons control locomotion — predicting and experimentally validating (by ablation + tracking) a role for the previously uncharacterized neuron PDB. Communicability and related walk-based measures summarize W spectrally. The "thermodynamic"/KMS-equilibrium program (Moutuou & Benali 2024, arXiv:2408.14221, published as Phys. Rev. Research 7, 033156, 2025; KMS states arXiv:2410.18222; Sunil, Benali & Moutuou 2026, arXiv:2604.02057) treats directed connectomes via graph C*-algebras and Kubo–Martin–Schwinger equilibrium states at inverse "temperature" β, producing a structure-derived functional connectivity in which β tunes the relative weight of short vs long paths (large β emphasizes short direct paths; smaller β allows longer/recurrent pathways). The 2026 paper compares this synaptic-derived functional layer directly to the extrasynaptic (neuropeptidergic) connectome. These models produce a functional or control graph from W alone and carry no explicit m.

**1.4 Phase-oscillator (Kuramoto) connectome models.**

  dθ_i/dt = ω_i + K Σ_j A_ij sin(θ_j − θ_i),  (1.5)

θ_i phase of neuron/region i, ω_i natural frequency, A_ij connectome adjacency, K global coupling. Used to study synchronization and cluster/symmetry structure on the C. elegans graph.

**1.5 Maximum-entropy / Ising models.**
The pairwise maximum-entropy (Ising) model for binary states σ_i ∈ {±1}:

  P(σ) = (1/Z) exp( Σ_i h_i σ_i + Σ_{i<j} J_ij σ_i σ_j ),  (1.6)

with h_i biases, J_ij couplings fit to match measured means and pairwise correlations; Z the partition function. Here J_ij is a *functional/effective* coupling, not anatomy. Recent multilayer work (arXiv:2509.20216, "Ising dynamics on multilayer networks with heterogeneous layers") studies Ising/Glauber dynamics on the C. elegans multiplex with distinct synaptic and extrasynaptic layers — directly relevant to layered J₀+J₁ decompositions.

**1.6 Data-driven latent dynamical models of worm activity.**
Kato et al. (2015, Cell 163:656–669) established that worm whole-brain activity is low-dimensional and embeds the motor-command sequence as a cyclic global trajectory. Morrison, Fieseler & Kutz (2021, Front. Comput. Neurosci. 14:616639, "Nonlinear Control in the Nematode C. elegans") fit a global linear model actuated by temporally sparse control signals (a data-driven control model). Connectome-constrained latent-variable models (Mi et al. 2022, ICLR) use a VAE whose generative prior is a connectome-constrained biophysical simulation, improving held-out neuron prediction over connectome-free models. Low-rank RNN theory (Mastrogiuseppe & Ostojic 2018, Neuron 99:609–623) writes recurrent connectivity as J = P + Σ_r m_r n_rᵀ/N (random plus low-rank), giving directly analyzable low-dimensional dynamics from connectivity — the natural object into which a fitted effective graph can be decomposed. Switching linear dynamical systems (rSLDS: Linderman et al. 2019, bioRxiv 621540, on Zimmer data; Costa, Ahamed & Stephens 2019, PNAS 116:1501–1510 adaptive locally-linear models; Costa et al. 2024, PNAS 121:e2318805121 Markovian behavior) approximate the flow as x_{t+1} = A_{z_t} x_t + b_{z_t} + noise with a discrete latent state z_t; each A_{z_t} is a state-conditioned effective graph — the closest classical analog to a modulator-indexed Jacobian, but the "state" is an abstract discrete label, not a measured modulator.

### 2. Neuromodulation insertion points (with mathematical form)

**(a) Gain modulation — diagonal, multiplicative.** Chance, Abbott & Reyes (2002, Neuron 35:773–782) showed background synaptic conductance multiplicatively scales the f–I curve slope (gain). In a rate model, replace f(u) with f(σ·u) or scale the transfer slope neuron-wise: x_i → f_i(u_i; σ_i). The Jacobian becomes

  J = D(σ) · W · diag(f′),  D(σ) = diag(σ_1,…,σ_n),  (2.1)

a **diagonal rescaling** of a base effective graph. Shine et al. (2018, eLife 7:e31130) found that increasing global neural gain drives an abrupt segregation→integration transition in a large-scale model, concluding that "neural gain modulation has the computational capacity to mediate the balance between integration and segregation in the brain." Li M et al. (2019, PLoS Comput. Biol. 15(10):e1006957 — first author Mike Li, not Li Y) linked whole-brain information-processing transitions to gain, showing that "the dynamics of the subcritical (segregated) regime are dominated by information storage, whereas the supercritical (integrated) regime is associated with increased information transfer (measured via transfer entropy)." Stroud et al. (2018, Nat. Neurosci. 21:1774–1783) showed targeted, per-neuron gain modulation of a fixed-connectivity RNN reshapes motor output and composes movement primitives — an explicit realization of the D(σ)J₀ family.

**(b) Weight reconfiguration / multilayer connectomes — additive.** Modulators add or remove functional edges: W(m) = W_syn + Σ_c m_c W_c, where W_c is the connectivity of modulator/receptor channel c and m_c its activation. This yields

  J(m) = J₀ + Σ_c m_c J_c,  (2.2)

the **additive-layer** hypothesis. Empirical W_c layers: Bentley et al. (2016, PLoS Comput. Biol. 12:e1005283) mapped monoamine and neuropeptide layers as a multiplex — the monoamine network has a highly disassortative, star-like topology with a rich-club of interconnected broadcasting hubs, and they report that most monoaminergic signalling occurs extrasynaptically (e.g., 100% of octopamine-receptor-expressing neurons receive no synaptic input from octomine-releasing neurons; 82%/76% for dopamine/serotonin). Ripoll-Sánchez et al. (2023, Neuron 111:3570–3589) built the neuropeptidergic connectome (from CeNGEN expression + Beets deorphanization), which "is characterized by high connection density, extended signaling cascades, autocrine foci, and a decentralized topology, with a large, highly interconnected core containing three constituent communities sharing similar patterns of input connectivity." Beets et al. (2023, Cell Reports 42:113058) provided system-wide peptide–GPCR interactions, biochemically mapping hundreds of concentration-dependent neuropeptide–receptor pairs (C. elegans encodes >300 neuropeptides from ~160 precursor genes and ~150 peptide GPCRs, per Taylor et al. 2021). Watteyne et al. (2024, Genetics 228:iyae141) review the structure-to-behavior peptide network. Conceptual grounding: Marder (2012, Neuron 76:1–11, "Neuromodulation of neuronal circuits: back to the future"; 2011, PNAS 108 Suppl 3:15542–15548 on variability/compensation/modulation) and Bargmann & Marder (2013, Nat. Methods 10:483–490) established that a fixed connectome supports many functional circuits depending on modulatory state.

**(c) Intrinsic-property modulation — changes to diagonal / bifurcation structure.** Modulators alter conductances and time constants (G_c, τ, adding/removing currents), changing the diagonal and curvature of J and thus the bifurcation regime (e.g., switching a neuron into bursting). In (1.1) this is m-dependence of G_c, E_j, or channel complement, moving the system across saddle-node/Hopf boundaries — the mechanism behind modulator-driven state switching in the stomatogastric tradition.

**(d) Mutually coupled neuronal–neurotransmitter systems.** Kringelbach et al. (2020, PNAS 117:9566–9576) close the loop of (0.1): a Hopf/dynamic-mean-field neuronal system is coupled to a neurotransmitter system via a receptor-density map (5-HT2A from PET), with the reverse coupling through a Michaelis–Menten release-and-reuptake equation driven by the source region's firing rate; the model's optimal fit depends causally on the empirical 5-HT2A map. Related: Deco et al. (2018, Curr. Biol. 28:3065–3074.e6, LSD/serotonin receptor-map model). Slow–fast/singular-perturbation analysis of (0.1): for ε→0, x tracks a critical manifold F(x;W,m)=0 while m drifts, and fast bifurcations of the layer system as m crosses thresholds produce state switching — the formal home of "modulator-driven state transitions." In trained RNNs, neuromodulation appears as learned gain/gating signals: Vecoven et al. (2020, PLoS ONE 15:e0227922); Stroud et al. (2018); Tsuda et al. (bioRxiv 2021; Neural Computation 2026) show neuromodulatory signals shift activity through "hyperchannels" to generate multiple context-relevant behaviors; and arXiv:2512.13859 (neuromodulation-inspired gated associative memory) uses peptide-like activity-dependent gating to extend memory capacity beyond the Hopfield limit.

### 3. Empirical results that constrain models

- **Randi et al. (2023, Nature 623:406–414; arXiv:2208.04790):** systematic optogenetic activation + simultaneous whole-brain calcium imaging over 23,433 neuron pairs across the head (n=113 animals) yields a functional atlas with sign, strength, temporal properties, and causal direction of signal propagation. Two constraints are decisive: (i) propagation **differs from anatomy-based predictions**, and mutant experiments show that "peptidergic extrasynaptic signaling contributes to neural dynamics by performing a functional role similar to that of a classical neurotransmitter," including dense-core-vesicle-dependent transients on sub-second-to-seconds timescales, sometimes with no wired connection; (ii) verbatim, "our measured signal propagation atlas better predicts the neural dynamics of spontaneous activity than do models based on anatomy… both synaptic and extrasynaptic signalling drive neural dynamics on short timescales." This is the primary target dataset for any J-estimator.
- **Atanas et al. (2023, Cell 186:4134–4151.e31):** brain-wide activity in freely-moving worms with probabilistic per-neuron encoding models; behavior/internal-state dependence of encoding across multiple timescales — direct evidence that effective coupling is state-dependent.
- **Dag et al. (2023, Cell 186:2574–2592.e20):** whole-brain dissection of the serotonergic system; six 5-HT receptors with distinct in-vivo kinetics (three core: MOD-1, SER-4, LGC-50), mapped to the connectome, with volume-transmission release from NSM — a concrete m-layer with receptor-specific timescales.
- **Flavell et al. (2013, Cell 154:1023–1035):** serotonin and neuropeptide PDF initiate/extend opposing behavioral states — foundational for modulator-driven state structure.
- **Fly effectome (Pospisil, Aragon, … Pillow 2024, Nature 634:201–209; bioRxiv 2023.10.31.564922):** an estimator for a linear dynamical model ẋ = Ax of the fly brain using stochastic optogenetic perturbation and instrumental variables, turning the connectome into a causal "effectome"; concludes dynamics are dominated by many small, near-independent circuits.

### 4. Where assumptions break

1. **Failure of timescale separation.** ε≪1 is false for fast peptidergic transients: Randi et al. document dense-core-vesicle signaling on sub-second timescales, so m is not adiabatically slow relative to x, and the slow-manifold reduction of (0.1) is invalid for those channels.
2. **Volume transmission is a field, not an edge.** Monoamine/peptide signaling diffuses; the correct object is a reaction–diffusion concentration field c(r,t) with receptor-weighted local read-out, only approximated by a static W_c edge. Spatial proximity (nerve-ring geometry) governs coupling, breaking the graph abstraction.
3. **Marder-style degeneracy.** Many parameter/modulator configurations produce indistinguishable activity (many-to-one map from (W,m) to dynamics). Hence (W,m) is generally not identifiable from activity alone; only certain functionals of J are.
4. **Absorption of unobserved m into history-dependence.** If m is unobserved and slow, marginalizing it makes the fitted fast dynamics **non-Markovian/history-dependent**: the apparent J acquires dependence on the trajectory's recent past (a history manifold) — exactly the reader's estimand. This is a feature, not a bug: the history dependence is the shadow of hidden modulatory state.
5. **Identifiability of (W,m) from J(x,m).** Since J = D_xF, one recovers a linearization, not the generative (W,m). Gain (D(σ)J₀) and additive (J₀+Σm_cJ_c) parameterizations can be observationally close; distinguishing them requires either perturbations (à la Randi/effectome) or strong structural priors.

### 5. "Kill search" — what already exists (novelty-assessment inputs)

**(i) State/modulator-dependent effective connectivity / Jacobians from learned models on worm (or cortical) data.**
- **Decomposed Linear Dynamical Systems (dLDS):** Yezerets, Mudrik & Charles (bioRxiv 2024.05.31.596903), published as Charles et al., "Decomposed Linear Dynamical Systems (dLDS)… reveal instantaneous, context-dependent dynamic connectivity in C. elegans," Communications Biology 2025, 8:1218 (method: Mudrik et al., arXiv:2206.02972). A learned generative model decomposing worm dynamics into recombinable linear operators — an explicitly **context/state-dependent dynamic connectivity**. This is the closest published prior art to a state-indexed effective-graph estimator, but it does not formally test the diagonal-gain vs additive-layer decomposition, and its "context" is not a measured modulator.
- **Connectome-constrained LDS fit to Randi's data:** Creamer, Leifer & Pillow (bioRxiv 2024.09.22.614271), a connectome-constrained noisy linear dynamical system fit to Randi's optogenetic perturbation data, evaluated by in-silico perturbation against held-out animals. Uses Randi as the fitting/validation target (item ii) but is a **single global** model — not state-dependent.
- **Score/diffusion approach:** a 2026 preprint (arXiv:2605.02852, "Score–Block Time Graphs"/SBTG; lead author reported as S. Kinger) reportedly uses denoising-score/diffusion models on worm whole-brain calcium imaging to recover lag-specific directed interactions described as Jacobians of the state-transition map, benchmarked against Cook/Randi/Bentley connectomes. *[Citation-status: EXISTENCE reported by secondary research; author list and specific claims NOT independently verified — treat as preliminary and read in full before relying on it.]*
- Classical rSLDS (Linderman 2019; Nassar 2019) and locally-linear models (Costa 2019) recover state-conditioned linear operators A_{z_t}, but with abstract discrete states, not measured m.

**(ii) Randi atlas as validation/benchmark.** Yes — Creamer/Leifer/Pillow and the SBTG preprint both use Randi's perturbation data/atlas as fit or validation, and Randi et al. themselves benchmark anatomy vs. atlas against spontaneous activity. No located *peer-reviewed* paper uses the atlas as a benchmark for a learned generative model **and** partitions by behavioral state.

**(iii) Structural form of state-dependence (D(σ)J₀ vs J₀+J₁ vs unstructured).** No located work — in worms or cortex — explicitly formulates and tests these three competing forms for how estimated effective connectivity changes across behavioral states. Cortical precedents exist for gain-based state changes (state-dependent gain modulation of recurrent networks) and for Jacobian-as-effective-connectivity (Deco et al. 2013, J. Neurosci. 33:11239), but not the comparative structural test. **This specific, modulator-indexed comparative test appears to be an open niche.**

### 6. Worked toy example: scalar gain driving a Hopf bifurcation in a 2-node network

Take an excitatory–inhibitory pair with self-excitation, a rotational coupling ω, gain g, and time constant τ:

  τ dx_E/dt = −x_E + g[ x_E − ω x_I ]
  τ dx_I/dt = −x_I + g[ ω x_E + x_I ]

Linearizing at the origin (or writing the linear network directly) gives the effective graph

  J(g) = (1/τ) [ [ g−1, −gω ], [ gω, g−1 ] ].

Eigenvalues: λ_± = (1/τ)[ (g−1) ± i gω ]. The real part Re λ = (g−1)/τ crosses zero at the **critical gain g\* = 1**, with imaginary part ±gω/τ ≠ 0 — the signature of a Hopf-type instability. For g<1 the origin is a stable focus (damped oscillation, frequency ≈ ω/τ); at g=1 the pair sits on the imaginary axis; for g>1 it is an unstable focus. Because the system is linear, g>1 gives unbounded growth; adding a saturating cubic term (the Stuart–Landau normal form, cf. (1.4)) converts this into a **supercritical Hopf bifurcation** with a stable limit cycle of radius ∝ √(g−1) and frequency ≈ ω/τ near onset.

Interpretation for the reader: a *single scalar neuromodulatory gain* g acting diagonally (here uniformly, D(σ)=g·I) moves the network across a Hopf boundary — the base graph J₀ = (1/τ)[[−1,−ω],[ω,−1]] is rescaled as J(g) = J₀ + (g−1)/τ · I in this uniform case, i.e., gain modulation is a **diagonal** perturbation of the effective graph. This is exactly the D(σ)J₀ hypothesis in miniature, and it shows how a modulator can switch a quiescent circuit into an oscillatory behavioral state without any change to the anatomical W.

## Recommendations

1. **Adopt the three-graph vocabulary explicitly** (W, J(x,m), functional) and state your estimand as J(x,m)=D_xF on a history manifold; this immediately differentiates you from connectome-only (J=W) and correlation-based (functional) work.
2. **Pre-register the three structural hypotheses** for state-dependence — H1 diagonal gain D(σ)J₀, H2 additive layer J₀+Σ_c m_c J_c, H3 unstructured change — and design your estimator to *nest* them (e.g., a low-rank-plus-diagonal parameterization J(σ) = D(σ)J₀ + Σ_c σ_c J_c). The kill search indicates this comparative test is your clearest novelty; make it the confirmatory core.
3. **Use Randi et al. (2023) as the causal validation target**, matching sign, strength, temporal profile, and causal direction — and explicitly benchmark against extrasynaptic (mutant) conditions. Because Creamer/Leifer/Pillow and possibly SBTG already fit Randi, differentiate by (a) state-indexing and (b) the structural-form test.
4. **Read three prior-art papers in full before claiming novelty**: dLDS (Commun. Biol. 2025, 8:1218), Creamer/Leifer/Pillow (bioRxiv 2024.09.22.614271), and SBTG (arXiv:2605.02852). Position your contribution as: modulator-indexed (not abstract-state) + structural-decomposition test + generative/score inference + Randi validation.
5. **Treat identifiability head-on.** Given Marder degeneracy and the J≠(W,m) gap, state clearly which functionals of J you claim to identify, and lean on perturbation data (Randi/effectome logic) or strong priors from the mapped W_c layers (Bentley/Ripoll-Sánchez) to break the gain-vs-additive ambiguity.
6. **Benchmarks that would change the plan:** if dLDS or SBTG is found to already test H1/H2/H3, pivot to the modulator-observability angle (linking σ to measured 5-HT/peptide reporters à la Dag 2023). If your estimator cannot separate H1 from H2 on synthetic data with realistic noise, add targeted perturbations rather than more free-behavior data.

## Caveats
- Several frontier items are **preprints** (Creamer/Leifer/Pillow; Linderman 2019; the SBTG score paper) and may change on peer review; the dLDS work is peer-reviewed (Commun. Biol. 2025).
- The SBTG paper (arXiv:2605.02852) is reported by secondary research only; its author list and specific claims were not independently verified. Do not cite it as established without reading it.
- 2026-dated arXiv IDs (e.g., 2604.02057, 2605.02852) are consistent with the current date but are recent and unrefereed.
- "c302", "The Virtual Brain", and Wilson–Cowan (1972) are described from general knowledge / secondary mentions; their primary citations were not independently verified in this pass (flagged below).
- The linear toy model's "Hopf" is exact only once a saturating nonlinearity is added; a purely linear network gives marginal (degenerate) oscillations at g\*=1.

## Citation status

**VERIFIED (publisher/PubMed/arXiv confirmed, directly or via targeted verification):**
- Randi, Sharma, Dvali, Leifer 2023, Nature 623:406–414, doi:10.1038/s41586-023-06683-4; arXiv:2208.04790.
- Kunert, Shlizerman, Kutz 2014, Phys. Rev. E 89:052805, doi:10.1103/PhysRevE.89.052805.
- Kunert-Graf, Shlizerman, Walker, Kutz 2017, Front. Comput. Neurosci. 11:53.
- Kunert, Maia, Kutz 2017, PLoS Comput. Biol. 13(1):e1005261; Kunert, Proctor, Brunton, Kutz 2017, PLoS Comput. Biol. 13(1):e1005303.
- Zhao et al. (BAAIWorm) 2024, Nat. Comput. Sci. 4:978–990, doi:10.1038/s43588-024-00738-w.
- Nicoletti et al. 2024, PLoS ONE 19(3):e0298105, doi:10.1371/journal.pone.0298105.
- Breakspear 2017, Nat. Neurosci. 20:340–352, doi:10.1038/nn.4497.
- Kringelbach et al. 2020, PNAS 117:9566–9576, doi:10.1073/pnas.1921475117; Deco et al. 2018, Curr. Biol. 28:3065–3074.e6, doi:10.1016/j.cub.2018.07.083.
- Yan et al. 2017, Nature 550:519–523, doi:10.1038/nature24056.
- Moutuou & Benali 2024, arXiv:2408.14221 (pub. Phys. Rev. Research 7, 033156, 2025); arXiv:2410.18222; Sunil, Benali, Moutuou 2026, arXiv:2604.02057.
- Ising multilayer 2025, arXiv:2509.20216; Bentley multilayer preprint arXiv:1608.08793.
- Kato et al. 2015, Cell 163:656–669, doi:10.1016/j.cell.2015.09.034.
- Morrison, Fieseler, Kutz 2021, Front. Comput. Neurosci. 14:616639, doi:10.3389/fncom.2020.616639; arXiv:2001.08332.
- Mi et al. 2022, ICLR (connectome-constrained LVM).
- Mastrogiuseppe & Ostojic 2018, Neuron 99:609–623, doi:10.1016/j.neuron.2018.07.003.
- Chance, Abbott, Reyes 2002, Neuron 35:773–782, doi:10.1016/S0896-6273(02)00820-6.
- Shine et al. 2018, eLife 7:e31130, doi:10.7554/eLife.31130.
- Li M et al. 2019, PLoS Comput. Biol. 15(10):e1006957, doi:10.1371/journal.pcbi.1006957 (first author Li M, not Li Y).
- Stroud et al. 2018, Nat. Neurosci. 21:1774–1783, doi:10.1038/s41593-018-0276-0.
- Marder 2012, Neuron 76:1–11, doi:10.1016/j.neuron.2012.09.010; Marder 2011, PNAS 108 Suppl 3:15542–15548, doi:10.1073/pnas.1010674108 (single author).
- Bargmann & Marder 2013, Nat. Methods 10:483–490, doi:10.1038/nmeth.2451.
- Bentley et al. 2016, PLoS Comput. Biol. 12(12):e1005283, doi:10.1371/journal.pcbi.1005283.
- Ripoll-Sánchez et al. 2023, Neuron 111:3570–3589.e5 (doi:10.1016/j.neuron.2023.09.043 — verify final DOI at proof; article confirmed).
- Beets et al. 2023, Cell Reports 42:113058, doi:10.1016/j.celrep.2023.113058.
- Watteyne et al. 2024, Genetics 228:iyae141, doi:10.1093/genetics/iyae141.
- Atanas et al. 2023, Cell 186:4134–4151.e31, doi:10.1016/j.cell.2023.07.035.
- Dag et al. 2023, Cell 186:2574–2592.e20, doi:10.1016/j.cell.2023.04.023.
- Flavell et al. 2013, Cell 154:1023–1035, doi:10.1016/j.cell.2013.08.001.
- Pospisil, Aragon, … Pillow 2024, Nature 634:201–209, doi:10.1038/s41586-024-07982-0; bioRxiv 2023.10.31.564922.
- Costa, Ahamed, Stephens 2019, PNAS 116:1501–1510, doi:10.1073/pnas.1813476116; Costa et al. 2024, PNAS 121:e2318805121.
- Charles et al. (dLDS) 2025, Communications Biology 8:1218, doi:10.1038/s42003-025-08599-3; bioRxiv 2024.05.31.596903; Mudrik et al. arXiv:2206.02972.
- Creamer, Leifer, Pillow 2024, bioRxiv 2024.09.22.614271, doi:10.1101/2024.09.22.614271.
- Vecoven et al. 2020, PLoS ONE 15(1):e0227922, doi:10.1371/journal.pone.0227922.
- arXiv:2512.13859 (neuromodulation-inspired gated associative memory).
- Cook et al. 2019, Nature 571:63–71; Varshney et al. 2011, PLoS Comput. Biol. 7:e1001066; Taylor et al. (CeNGEN) 2021, Cell 184:4329–4347.

**UNVERIFIED / FLAGGED (could not independently confirm primary citation this pass):**
- SBTG score/diffusion worm paper, arXiv:2605.02852 — existence reported by secondary research; author list/claims not independently verified.
- OpenWorm c302 primary reference (Gleeson et al.) — not independently verified.
- The Virtual Brain primary reference (Sanz-Leon et al.) — not independently verified.
- Wilson & Cowan 1972 (Biophys. J. 12:1–24) — foundational, cited from general knowledge, not re-verified this pass.
- Linderman et al. 2019 "Hierarchical recurrent state space models…" — bioRxiv 621540; confirmed as preprint via secondary research, published version not confirmed.
- Tsuda et al. — bioRxiv 2021.05.31.446462; Neural Computation 2026 version reported by secondary research, not independently verified.
- Deco et al. 2013, J. Neurosci. 33:11239 — cited from secondary research (subagent), page/volume not independently confirmed this pass.

## Novelty assessment (established vs open)

**Established (do not claim as novel):** (1) that a fixed connectome supports many functional circuits under neuromodulation (Marder; Bargmann–Marder); (2) that anatomy under-predicts function and extrasynaptic signaling matters on seconds timescales (Randi); (3) that gain modulation reconfigures network activity/dynamics as a diagonal operation (Chance–Abbott–Reyes; Stroud; Shine; Li); (4) that learned dynamical/generative models can recover Jacobian-like, context-dependent effective connectivity from worm whole-brain data (dLDS; connectome-constrained LDS; likely SBTG); (5) that Randi's data can serve as a causal fitting/validation target (Creamer/Leifer/Pillow; effectome logic in fly).

**Apparently open (candidate novelty, stated with uncertainty):** a *modulator-indexed* (not merely abstract-discrete-state) estimator of the point-wise Jacobian J(x,m) that (a) is fit via score-based/generative inference on a history manifold, (b) is validated against the Randi propagation atlas including extrasynaptic mutant conditions, and (c) **formally tests** whether behavioral-state changes act as a diagonal gain D(σ)J₀, an additive layer J₀+Σ_c m_c J_c, or unstructured change. No located work does all three; item (c) in particular appears untested in worms or cortex. Uncertainty: the dLDS and SBTG papers are close enough that a full reading is required before asserting novelty, and the SBTG preprint is itself unverified.