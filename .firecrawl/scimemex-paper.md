# Breaking Bored? SciMemeX: Towards Automatic Generation of Scientific Memes from Research Articles

**Sandeep Kumar, Asif Ekbal**

Department of Computer Science and Engineering, Indian Institute of Technology Patna, India
(sandeep_2121cs29,asif)@iitp.ac.in

## Abstract

The rapid growth of scientific literature makes
it increasingly difficult to communicate a paper’s core novelty in a concise and accessible manner. While prior work has explored
visual summaries such as graphical abstracts,
emerging informal formats such as scientific
memes remain largely unstudied from a computational perspective. Scientific memes rely
on concise visual–textual juxtaposition and
contrast to highlight how new work differs
from prior approaches, making them a challenging test case for controlled abstraction,
contrastive reasoning, and creative compression in generative models. We formalize
scientific meme generation as a novel multimodal, contrastive generation task and propose
*SciMemeX*, a modular multi-agent framework
that extracts core innovations, selects contextually appropriate meme templates, and iteratively generates and refines captions through
contrastive feedback-guided refinement. We
evaluate generated memes along three complementary dimensions—*scientific fidelity*,*clar-*
*ity*, and*engagement potential*—and conduct
extensive automatic and human evaluations
on 1.5k computer science papers. Our results show that SciMemeX consistently outperforms strong single-agent prompting and existing multi-agent baselines across all evaluation
dimensions, suggesting that scientific meme
generation provides a constrained setting for
exploring controlled creative generation in scientific communication context. We will make
the dataset and code public¹.

## 1 Introduction

Scientific progress relies not only on the discovery
of new knowledge, but also on the effective communication of research contributions to audiences
(Leshner,2003;Marín-González et al.,2017). As
the volume of scientific publications continues to

grow, researchers face increasing difficulty in conveying the novelty and significance of their work
in a concise and accessible manner (Warren,2003).
To address this challenge, the scientific community
has developed a range of compressed and visual
communication formats, including abstracts, short
talks, posters, and graphical or visual abstracts.
Prior studies show that such visual formats can substantially increase online visibility and engagement
with scientific articles, particularly on social media
platforms (Oska et al.,2020). In parallel, platforms
such as Twitter (X) and LinkedIn have become
important channels through which researchers contextualize and disseminate their works beyond traditional academic venues (Bik and Goldstein,2013;
Van Eperen and Marincola,2011).

Within this evolving communication landscape,
*memes*have emerged as a widely used form of
digital expression characterized by concise visual–
textual juxtaposition, contrast, and often humor.
Prior work in education and communication studies
shows that meme-based artifacts can support learning, recall, and engagement, and are increasingly
viewed as a stable form of online discourse and
digital literacy (Riser et al.,2020;Shibilski,2021;
Myers,2022).Recent practitioner accounts further
document the use of science memes in conference
presentations, teaching, blogs, and social-media
outreach, suggesting that memes are increasingly
being explored as informal science communication tools (Joseph,2023;Tran,2024).These findings suggest that memes may plausibly function
as lightweight, contrastive explanatory artifacts in
informal or semi-formal scientific communication
settings.Despite these emerging uses,*scientific*
*memes*remain largely unstudied from a computational perspective. There is currently no systematic
framework that formalizes what constitutes a scientific meme, how such artifacts encode a paper’s conceptual novelty, or whether generative models can
produce them while preserving scientific fidelity.

<sup>1</sup>https://anonymous.4open.science/r/
SciMeme-5FE1/

---

Rather than assuming memes to be a universally
validated dissemination medium, we treat them as
an*emerging artifact class*whose structure and generative feasibility warrant systematic investigation.

Unlike graphical abstracts, which aim for broad
coverage of a study’s content, scientific memes
focus on contrastive compression—isolating and
juxtaposing what is new against what came before.
Importantly, we do not treat scientific memes as
general-purpose internet humor artifacts. Rather,
we target domain-literate scientific communication, where humor is optional and contrastive novelty representation remains the core structural requirement.Following prior work that treats internet memes as template-based, genre-like artifacts shaped by recognizable conventions (Shifman,2013;Wiggins and Bowers,2015), we focus on a contrastive “before vs. after” / “prior vs.
proposed” structure, which highlights how a new
paper differs from prior approaches.This format
aligns with well-established cognitive principles,
including dual coding theory (Paivio,1990), contrastive framing effects (Kahneman and Tversky,
1984), and the role of surprise in memorability
(Górka,2014). By juxtaposing prior limitations
with new contributions in a compact visual–textual
form, such memes offer a concise representation
of scientific novelty well suited to fast-paced digital environments. Beyond its application context,
scientific meme generation poses challenges that
are independently relevant to natural language processing and multimodal generation. Producing a
faithful scientific meme requires models to identify
a paper’s central contribution, reason contrastively
with respect to prior work, compress this distinction
into a highly constrained visual–textual format, and
balance creativity with factual correctness. These
requirements go beyond standard summarization or
captioning tasks, instead demanding controlled abstraction, contrastive reasoning, and creative compression—capabilities that remain challenging for
current large language models. As such, scientific
meme generation serves as a compact testbed for
studying these fundamental capabilities, regardless

of the eventual adoption of memes as a dissemination medium.

In this paper, given a scientific paper, our goal is
to automatically generate a meme that accurately
and clearly communicates the paper’s key novelty
relative to prior work while remaining faithful to
the underlying scientific content. To this end, we
introduce**SciMemeX**, an agentic framework for
generating scientific memes directly from research
papers. SciMemeX integrates multiple specialized
agents that collaboratively extract core contributions, select contextually appropriate meme templates, and generate contrastive captions through
an iterative refinement process.

Our contributions are*threefold*:**(i)**we formalize
scientific meme generation as a novel multimodal,
contrastive generation task that probes controlled
abstraction and creative compression in generative
models;**(ii)**we propose**SciMemeX**, an agentic
framework that decomposes the task into contribution extraction, template selection, and iterative
contrastive feedback-guided meme generation; and
**(iii)**through comprehensive automatic and human
evaluations, we show that our approach consistently outperforms single-agent prompting and existing multi-agent baselines in terms of scientific
fidelity, clarity, and engagement.

## 2 Related Work

Prior work on scientific communication has explored a range of textual (Hermann et al.,2015;
Sefid and Giles,2022;Sharma et al.,2024;Dong
et al.,2021) and multimodal abstractions, including
summaries (Tan et al.,2025;Kumar et al.,2024a),
graphical abstracts (Kawada et al.,2025), posters
(Wang et al.,2024b;Tanaka et al.,2024;Landis
and Duscher,2022) , and slide-style presentations
(Sefid et al.,2021), aimed at conveying a paper’s
core contributions in a concise and accessible form.
More recently, P2P (Sun et al.,2025) used coordinated LLM agents to generate poster-style visual
summaries of papers. However, such methods organize paper content rather than foreground novelty in brief, engaging formats, motivating more
concise communication artifacts. Prior work on
automatic meme generation has largely targeted
general-purpose social media content, emphasizing
humor and virality over factual or technical accuracy (Vyalla and Udandarao,2020;Wang and Lee,
2024a).

While recent vision language models enable more context aware meme generation, they primarily target general purpose social media content and
do not account for scientific fidelity or the need to
distinguish new contributions from prior work. To
the best of our knowledge, no prior work addresses
scientific meme generation as a dedicated scientific
communication task. We hope this work inspires
future research at the intersection of AI-driven creativity and accessible science dissemination.

## 3 Methodology

## 3.1 Problem Definition

We define scientific meme generation as the task of
producing a meme that communicates a paper’s key
novelty relative to prior work. Let a paper be *p* (abstract, sections), a template set *M*, and a caption
space *C*. A meme is *m*= (*t,c*) with *t∈M*, *c∈C*,
contrasting prior work and new contributions. A
team of agents*A*=*{* Concisio, Template_Selector,
Generator, Critics*}* induces a policy *π*<sub>A</sub>(*m|p*)
realized through an exploration and exploitation
contrastive feedback loop. Rather than formulating
a directly solved optimization problem, we define a
conceptual objective that guides the iterative generation and refinement process. In practice, candidate
memes are assessed along three desiderata, namely
scientific fidelity, clarity and interpretability, and
engagement. These criteria are used by the critique
agents to compare and refine candidates during the
search process. The final output corresponds to
the best candidate identified through this feedback
driven procedure.

$$
m=(t,c)
$$

$$
t\in\mathcal{M},c\in\mathcal{C}
$$

$$
\pi_{\mathcal{A}}(m\mid p)
$$

Abstract
Other Sections
Input

Concisio Agent
Innovative Reflections
TSA
Meme Text Generator

K Generated Memes
ImageFip/
Multimodal LLM
Meme

Exploration-
Exploitation loop

Best Meme
Worst Meme

Ranking
Scores and feedbacks

Critique Agent

Continitive Feedback-Guided Meme Text Generator

Judges

Figure 1: Agentic framework for meme generation. The
process consists of an initial**exploration**phase (diverse
template selection and generation) followed by iterative
**exploitation**through contrastive refinement, guided by
critique feedback.

## 3.2 Framework Overview

Grounded in the dual perspective of human
creative cognition and computational exploration–exploitation, our methodology operationalizes this cycle within an agentic computational

framework. SciMemeX instantiates alternating
phases of divergent ideation and convergent critique through a coordinated ensemble of agents
that specialize in generation, selection, and evaluation. Each agent embodies a cognitive role—idea
condensation, template prioritization, caption generation, or multi-axis critique—collectively forming an adaptive reasoning loop. This design
unites psychological models of creativity (Guilford,1967;Torrance,1988;Sawyer,2012) with
algorithmic principles of adaptive search (Sutton
and Barto,2018;Kaelbling et al.,1996), framing scientific meme generation as both a cognitive
simulation and an optimization process (Dawkins,
1976;Blackmore,1999). The workflow of the
proposedSCIMEMEXframework is illustrated in
Figure1, with the detailed algorithm provided in
Appendix1. The input consists of a paper’s abstract and, optionally, its other sections. If only
the abstract is available, the*Concisio Agent*is bypassed; otherwise, the*Concisio Agent*summarizes
the paper and extracts key innovative reflections,
which are then passed to the*Template Selector*
*Agent (TSA)*. The TSA identifies suitable meme
templates and forwards them to the*Meme Text*
*Generator*, which produces *k* captioned meme candidates. Each candidate is evaluated independently
by three*Critique Agents*—*Fidelity Critique*,*Clar-*
*ity Critique*, and*Engagement Critique*—whose
scores are used to rank the memes. The best- and
worst-performing memes are then provided to the
*Contrastive Feedback-Guided Generator*, which refines new captions by integrating both positive and
negative feedback. This exploration–exploitation
loop repeats until convergence or a predefined iteration limit.Finally, the selected template–caption
pair is rendered into the final meme image.

In the following sections, we describe the principal components of**SciMemeX**, including the
meme-generation agents, their role specifications,
and the inter-agent communication strategy. The
proposed multi-agent system operates within an
exploration–exploitation loop: agents first extract
a contrastive summary from the input paper *p*, select suitable templates, generate captioned meme
candidates, and iteratively refine them based on
critique feedback until convergence or a predefined computational budget is reached. These
meme-generation agents—*Concisio*,*Template Se-*
*lector*,*Generator*, and*Critique*—form the core
of our framework, collectively enabling adaptive
reasoning and creative synthesis across iterative refinement cycles.SCIMEMEXis a training-free
inference-time framework that coordinates specialized LLM agents via contrastive comparison and
feedback-guided refinement.

## 3.3 Concisio Agent

The Concisio Agent integrates comparative analysis and concise synthesis into a reasoning module.
Given a research paper *R*, the agent first performs
a comparison between prior and current methodologies². It identifies a set of prior approaches
*M*<sup>old</sup>=*{M*<sup>p</sup><sup>1</sup>*,...,M*<sup>p</sup><sup>n</sup>*}* and extracts the main
3
contribution(s) of the current work as *M*<sub>new</sub>. For
each *M*pi*∈M*old, the agent identifies conceptual
and technical distinctions ∆ibetween *M*newand
*Mpi*, capturing differences in mechanism, objective, or theoretical framing. Based on this comparison, the agent produces a compact summary *S<sup>∗</sup>*
from *M*<sub>old</sub>, *M*<sub>new</sub>, and the set of distinctions *{*∆<sub>i</sub>*}*.
The summary *S<sup>∗</sup>* captures (i) how the problem was
addressed previously, (ii) what the current paper
does differently, and (iii) why this difference is
meaningful.We operationalize a paper’s*key contri-*
*bution*as contrastive novelty relative to prior work:
what prior approaches did, what the current work
changes, and why the difference matters.The output is organized into three interpretable segments,
*Prior Work (Old Ideas)*,*Proposed Approach (New*
*Idea)*, and*Core Differences*, and expressed as a concise narrative of approximately 100 words. This
agent provides a clear and faithful foundation for
downstream modules such as the Template Selector
and Meme Generator.

$$
\mathcal{R}
$$

$$
\mathcal{M}_{\mathrm{old}}=\left\{M_{p_1},\ldots,M_{p_n}\right\}
$$

$$
M_{new}
$$

$$
(s)^{3}
$$

$$
M_{p_{i}}\in\mathcal{M}_{old}
$$

$$
\Delta_{i}
$$

$$
M_{new}
$$

$$
M_{p_{i}}
$$

$$
S^{*}
$$

$$
\{\Delta_{i}\}
$$

$$
\mathcal{M}_{old},M_{new}
$$

$$
S^{*}
$$

## 3.4 Template Selector Agent

The Template Selector Agent (TSA) is a prompt
based LLM module that uses the innovation summary *S*<sup>∗</sup>, consisting of the abstract and extracted
reflections, to identify suitable meme templates
from the set *M*.We define *M* as a fixed, externally curated pool of 250 meme templates collected
from publicly available meme-template repositories and websites, including ImgFlip<sup>4</sup> and other
open meme-template catalogs, after filtering.We

$$
S^{*}
$$

retain templates containing at least two editable text
placeholders and a clear visual structure suitable
for contrastive framing, while excluding singlecaption, visually unclear, or non-contrastive “prior
work vs. proposed work” formats. Given *S*<sup>∗</sup>, the
agent selects a top *k* subset *T*<sub>p</sub>*⊆M* that is appropriate for highlighting the contrast between prior
work and the new contribution.

$$
S^{*}
$$

$$
T_{p}\subseteq\mathcal{M}
$$

## 3.5 Meme Generation and Refinement

As illustrated in Figure1,SCIMEMEXconducts meme generation through two complementary stages—initial exploration and feedbackguided refinement—forming an adaptive exploration–exploitation loop grounded in both human
creative cognition and algorithmic search.

**Initial Exploration.**Conditioned on the innova-
∗
tion summary *S* and the selected template set *T*<sub>p</sub>,
the Meme Text Generator produces *k* captioned
candidates per template. This phase emphasizes
diversity and breadth, mirroring human*brainstorm-*
*ing*: numerous unconstrained captions are generated to cover varied humorous framings and contrasts between prior and new work. The resulting
candidates form the initial pool for evaluation.

$$
T_{p},
$$

$$
S^{*}
$$

**Feedback-Guided Refinement.**Each candidate
is scored by the critique agent, which also provides textual feedback. The highest- and lowestranked memes (*m<sup>+</sup>*, *m<sup>−</sup>*) serve as contrastive anchors for the Feedback-Guided Meme Generator,
which synthesizes a new batch of *k* refined captions
that retain the strengths of *m<sup>+</sup>* while avoiding the
weaknesses of *m<sup>−</sup>*. Although this stage primarily exploits prior feedback, it preserves controlled
exploration via stochastic prompting and stylistic
variation. The loop iterates until convergence or a
budget cap, returning the highest-scoring candidate
⋆
*m*.

$$
(m^{+},m^{-})
$$

$$
m^{+}
$$

$$
m^{-}
$$

$$
m^{\star}
$$

Finally, the selected meme template and generated caption are rendered into the final meme image.
We use ImgFlip as the default renderer whenever
the selected template is available in its public catalog; this accounts for 86% of the generated memes
in our evaluation. For the remaining 14%, where
the selected template is not available in ImgFlip
or requires a custom multi-panel composition, we
use GPT-4o as the fallback renderer. We selected
this multimodal renderer once at implementation
time based on pilot checks for prompt following,
template adherence, and readable text placement;

<sup>2</sup>Concisio relies only on information explicitly provided
in the input paper rather than external model knowledge, reducing the risk of hallucinated prior-work comparisons.

<sup>3</sup>Many papers contain multiple complementary contributions. In such cases, Concisio extracts multiple contrastive
conceptual and technical distinctions relative to prior work
and synthesizes the most salient innovations into a unified
compact summary.

4
https://imgflip.com/ the renderer is not chosen on a per-paper basis. Importantly,SCIMEMEXoutputs a template–caption
pair, so the agentic generation pipeline is renderingagnostic.During iterative refinement, we intentionally keep the meme template fixed and refine only
the caption. This design ensures that critique feedback remains grounded in a stable visual framing;
changing templates mid-loop would invalidate the
critics’ judgments and led to unstable or incoherent
revisions in pilot experiments.

## 4 Evaluation

A central challenge in this work is to ensure that
the proposed evaluation framework is both theoretically sound and practically reliable. To this end, we
designed a three-phase evaluation pipeline that progressively integrates expert insights, LLM-as-judge
validation, and final human benchmarking. This
pipeline was constructed to address common criticisms of evaluation in generative tasks, namely: (i)
overreliance on ad hoc human judgments, (ii) the
limited generalizability of traditional metrics (e.g.,
BLEU/ROUGE), and (iii) the lack of systematic
validation for LLM-based scoring.

## 4.1 Phase I: Expert-Driven Metric Design

We began with a structured pilot study involving twelve senior experts, including mid-career
faculty, postdoctoral researchers, and PhD candidates (5–10 years of experience) with more than
5+ ACL/ICLR or related publications. Experts
evaluated memes generated for a representative
set of 80 papers drawn from their own publication history and from comparable peers. Rather
than limiting themselves to one-shot generations,
each expert engaged in an interactive human-inthe-loop process: they prompted the LLM of
their choice (CHATGPT,GEMINI, orCLAUDE)
(Achiam et al.,2023;Anthropic,2023;DeepMind,
2023), inspected its outputs, and iteratively refined
the memes with targeted feedback until they were
satisfied that the result captured the paper’s core
contribution. Across four workshop rounds, experts not only judged the resulting memes but also
critiqued and revised them, using the iterative process itself as a way to stress-test evaluation criteria.
In doing so, they revealed both the strengths and
limitations of LLMs when steered by human expertise, and distilled a set of evaluation dimensions
that remained robust even under this adversarial,
feedback-driven setting.

We observed consistent failures of lexical overlap metrics (e.g.,BLEU,ROUGE) (Papineni et al.,
2002;Lin,2004) and vision–language similarity
scores (Radford et al.,2021a) to capture the desiderata of our task. Through structured annotation sessions and consensus-building (Artstein and Poesio,2008a;Krippendorff,2018), experts distilled
three dimensions as essential and non-redundant:
(i)**Scientific Fidelity**, (ii)**Clarity & Interpretabil-**
**ity**, and (iii)**Engagement Potential**. Importantly,
the design process reached saturation: after four
workshop rounds, no additional dimensions were
proposed. Scientific Fidelity and Engagement Potential were each operationalized on a**1–5 Likert**
**scale**, allowing fine-grained differentiation. In contrast, Clarity & Interpretability employed a**3-point**
**scale**, as pilot testing showed that annotators could
not reliably sustain more granular judgments in
this dimension without over-interpretation. We discuss the details of the evaluation metrics and their
definitions in the appendixC.

## 4.2 Phase II: LLM-as-Judge Validation

Having formalized the rubrics, we implemented
them as structured prompts and evaluated their reproducibility under an LLM-as-judge paradigm
(Cai et al.,2025) using GPT-4.1 (Achiam et al.,
2023) on a new sample of 300 memes. Six independent domain experts (senior PhD students and
early-career researchers, not involved in Phase I)
annotated each meme on all three dimensions, with
three raters per item.

Inter-annotation agreement was measured using
standard reliability metrics (Artstein and Poesio,
2008a;Krippendorff,2018), yielding substantial
human–human consistency (*τ*= 0*.*69, *κ*= 0*.*66,
*r*= 0*.*72, *α*= 0*.*68). GPT-4.1, prompted with
the same rubric under controlled settings, was
then applied to the identical dataset. Its ratings
showed high concordance with human judgments
(*τ*= 0*.*71, *κ*= 0*.*68, *r*= 0*.*74). These values fall
within or above commonly accepted thresholds for
strong inter-annotator reliability in computational
linguistics (Artstein and Poesio,2008b;Krippendorff,2018). Importantly, the LLM-as-judge framework achieved higher internal consistency than additional human raters with limited domain expertise, reinforcing its validity as a scalable proxy.
Prompt templates, Annotator Guidelines, Annotation Training, Annotation Steps are discussed in
detail AppendixE.

$$
(\tau=0.71,\kappa=0.68,r=0.74)
$$

---

|  | Method | Fidelity | Clarity | Engagement | Total |
| --- | --- | --- | --- | --- | --- |
| Single-Agent Baselines (CoT Prompting) |  |  |  |  |  |
| Mistral-7B (I) | CoT | 2.84 | 1.70 | 3.18 | 7.72 |
| Llama-3.1-8B (I) | CoT | 3.02 | 1.77 | 3.30 | 8.09 |
| Llama-3.1-70B (I) | CoT | 3.12 | 1.82 | 3.47 | 8.41 |
| Claude-3.5 Sonnet | CoT | 3.25 | 1.86 | 3.60 | 8.71 |
| Gemini-1.5 Pro | CoT | 3.34 | 1.89 | 3.72 | 8.95 |
| GPT-4o | CoT | 3.48 | 1.93 | 3.85 | 9.26 |
| Template-Based Meme Systems |  |  |  |  |  |
| Memeify | Template-Captioning | 2.21 | 1.41 | 2.96 | 6.58 |
| MemeCraft | Template-Captioning | 2.48 | 1.55 | 3.12 | 7.15 |
| Multi-Agent Frameworks |  |  |  |  |  |
|  | MAD Self-Reflect | 3.17 3.32 | 1.85 1.92 | 3.56 3.68 | 8.58 8.92 |
| Llama-3.1-70B (I) | ChatEval | 3.45 | 1.95 | 3.77 | 9.17 |
|  | Self-Refine |  |  |  |  |
|  | SciMemeX (Ours) | 3.56 | 1.97 | 3.74 | 9.27 |
|  |  | 4.31 | 2.09 | 3.90 | 10.30 |
|  | MAD Self-Reflect | 3.26 | 1.87 | 3.67 | 8.80 |
| Claude-3.5 Sonnet | ChatEval | 3.41 3.53 | 1.94 | 3.75 | 9.10 |
|  | Self-Refine |  | 1.97 | 3.83 | 9.33 |
|  | SciMemeX (Ours) | 3.67 | 2.00 | 3.82 | 9.49 |
|  |  | 4.44 | 2.12 | 3.96 | 10.52 |
| Gemini-1.5 Pro | MAD | 3.34 | 1.90 | 3.72 | 8.96 |
|  | Self-Reflect | 3.54 | 1.98 | 3.78 | 9.30 |
|  | ChatEval | 3.67 | 2.00 | 3.83 | 9.50 |
|  | Self-Refine | 3.79 | 2.03 | 3.89 | 9.71 |
|  | SciMemeX (Ours) | 4.52 | 2.14 | 4.00 | 10.66 |
|  | MAD | 3.52 | 1.94 | 3.85 | 9.31 |
| GPT-4o | Self-Reflect | 3.70 | 2.00 | 3.88 | 9.58 |
|  | ChatEval | 3.83 | 2.03 | 4.04 | 9.90 |
|  | Self-Refine | 3.94 | 2.04 | 4.02 | 10.00 |
|  | SciMemeX (Ours) | 4.66 | 2.16 | 4.12 | 10.94 |

Table 1: Automatic evaluation results across six open- and closed-source models released by July 2025. We
report mean scores for Fidelity, Clarity, Engagement, and total composite scores for both single-agent (CoT) and
multi-agent frameworks. “I” denotes instruction-tuned variants (Instruct models).

## 4.3 Phase III: Final Human Evaluation

The final phase benchmarked our framework
against three closely relatedbaselines (MAD, Chat-
Eval, and Self-Refine)through a targeted human
evaluation. We sampled 250 representative papers
spanning diverse subfields (language modeling, machine translation, multimodal reasoning, and interpretability). To ensure consistency, we employed
the same expert annotators from Phase II, who were
already calibrated on the rubric but remained blind
to system identity. All memes were rated on Fidelity (1–5), Clarity & Interpretability (1–3), and
Engagement Potential (1–5), with human–human
reliability monitored using Kendall’s *τ*, Cohen’s *κ*,
and Krippendorff’s *α*, and raw agreement reported
without adjudication. Meme order was randomized
within each paper to prevent bias, enabling a fair,
head-to-head comparison of our framework against
the baselines. We discuss the annotator’s pay in
AppendixE.1. This also serves as an additional
safeguard against potential bias in LLM-based evaluation, as annotators were blind to system identity.

## 5 Experiments

We evaluateSCIMEMEXon 1.5k computer
science papers randomly sampled from arXiv
AI/NLP/ML categories from 2022–2023, using a
30/70 validation–test split. Papers were sampled
uniformly from these categories, and we excluded
only papers whose PDFs could not be parsed into
usable title, abstract, and main-body text by GRO-
BID (Developers,2008). Automated evaluation
is conducted on the held-out test split using the
LLM-as-judge rubric validated in Section4.2. We
compare against single-agent prompting, templatebased meme systems, and multi-agent baselines,
including CoT, Memeify, MemeCraft, MAD, Self-
Reflect, ChatEval, and Self-Refine. All baselines
are evaluated with the same task inputs and the
same underlying backbone model where applicable; full implementation and baseline details are
provided in AppendicesAandB.

## 5.1 Results

## 5.1.1 Automatic Evaluation

We report the results ofSCIMEMEXand baselines
using the automated metrics (discussed in4.2) in

---

Table1. We find thatSCIMEMEXachieves the
best scores across all metrics. The largest and
most stable improvements appear in*Scientific Fi-*
*delity*(+0*.*72–+0*.*77 over the strongest baseline),
confirming that our feedback-driven generation
better preserves the underlying claims of the paper.*Clarity*also improves consistently (+0*.*11–
+0*.*12), indicating that structured refinement enhances linguistic precision and interpretability.*En-*
*gagement*shows smaller, yet positive gains (+0*.*10–
+0*.*16), suggesting that faithfulness and readability
are not achieved at the cost of appeal. Templatebased meme systems perform substantially worse
across all metrics, reflecting the limitations of fixed
template–captioning for scientific content. These
improvements are consistent across model capacities—from Llama-70B to GPT-4o—demonstrating
thatSCIMEMEXgeneralizes beyond a specific architecture. In contrast toCHATEVALandSELF-
REFINE, which perform unguided multi-round
refinements, our approach employs a structured
exploration–exploitation strategy that explicitly
contrasts the best and worst generations in each
round. This contrastive critique–feedback loop enables targeted improvement across iterations rather
than generic self-revision, allowing the model to
progressively converge toward high-fidelity, interpretable, and engaging memes.

## 5.1.2 Human Evaluation

We now report the final human evaluation comparingSCIMEMEXagainst three strong baselines.
Full details of the protocol are provided in Section4.3. Table2summarizes mean human scores⁵
with 95% confidence intervals, estimated via paired
bootstrap resampling (1,000 draws over paper-level
items). Statistical significance was assessed using paired bootstrap with Holm correction across
metrics.SCIMEMEXconsistently outperforms all
baselines on all three dimensions. Improvements
are significant for*Fidelity*(*p<.*01) and*Clarity &*
*Interpretability*(*p<.*05), while*Engagement Po-*
*tential*also shows a significant gain (*p<.*05), indicating that higher factual precision and clarity
are achieved without compromising creative appeal. After normalization,SCIMEMEXachieves
the highest composite score (2*.*43), reflecting a
notable improvement of +0*.*43 over the strongest
baseline.To verify that agreement was not driven
by midpoint clustering, we report Phase III human

score distributions in AppendixG.

## 5.2 Analysis

12Fidelity
Clarity
Engagement
10Total
8
Score6
4
2
0 0 1 2 3 4 5 6
Iteration

Figure 2: Stacked plot of scores across iterations. The
black line shows the total score, while the stacked areas indicate the component contributions from Fidelity,
Clarity, and Engagement.

## 5.2.1 Effect of Iterative Refinement

Figure2shows the progression of scores across successive critique–feedback iterations. We observe
consistent improvements across all three evaluation dimensions during the early stages, with gains
stabilizing after approximately the third iteration.
*Scientific Fidelity*rises from 3.70 at iteration 0 to
4.76 by iteration 3 (+28.6%), reflecting improved
preservation of core scientific claims through targeted refinement.*Clarity & Interpretability*improves from 2.00 to 2.19 (+9.5%), while*Engage-*
*ment Potential*increases from 3.88 to 4.18 (+7.7%),
showing that factual precision does not come at the
cost of communicative appeal. Overall, the composite score improves from 9.58 to 11.13 (+16.1%),
indicating that most benefits of iterative feedback
occur within the first few rounds. This does not
represent a hard architectural limit: the framework
does not impose a fixed iteration cap, and further
gains may arise from expanding the template space
or critique signals. We discuss a refinement case
study in AppendixM.

## 5.2.2 Template Diversity Analysis

Automatic meme generation systems often suffer
from*template collapse*, repeatedly relying on a
small number of formats. We therefore analyze
template diversity on the Phase III evaluation set
using Unique Template Count (UTC), template entropy (Shannon,1948), and Top-1 template frequency (Holtzman et al.,2020). As shown in Table3,SCIMEMEXexhibits higher diversity than
single- and multi-agent baselines, with higher entropy and lower template dominance. This improvement arises from a single, contrast-aware template
selection step; removing the Template Selector

<sup>5</sup>From the 250 papers sampled to construct a balanced
pool, we selected a 100-paper subset.

---

| System | Fidelity (1–5) | Clarity (1–3) | Engagement (1–5) | Norm. Total (0–3) |
| --- | --- | --- | --- | --- |
| MAD3 | . 48 ± 0 . 11 | 1 . 92 ± 0 . 07 | 3 . 83 ± 0 . 12 | 1 . 78 |
| ChatEval3 | . 84 ± 0 . 10 | 2 . 03 ± 0 . 06 | 4 . 04 ± 0 . 11 | 1 . 97 |
| Self-Refine3 | . 97 ± 0 . 10 | 2 . 05 ± 0 . 06 | 4 . 05 ± 0 . 11 | 2 . 00 |
| SciMemeX (Ours) | ⋆⋆ 4 . 68 ± 0 . 09 | ⋆ 2 . 16 ± 0 . 05 | ⋆ 4 . 21 ± 0 . 09 | ⋆⋆ 2 . 43 |
| ∆vs. best baseline+0 | . 71 | +0 . 11 | +0 . 16 | +0 . 43 |

$$
1.92\ _{\pm0.07}
$$

$$
3.48\ _{\pm0.11}
$$

$$
3.83\ _{\pm0.12}
$$

$$
3.84\pm\substack{0.10}
$$

$$
2.03\pm0.06
$$

$$
\begin{array}{c}{3.01\pm0.10}\\ {3.97\pm0.10}\\ \end{array}
$$

$$
4. 0 4 _ {\pm 0. 1 1}
$$

$$
3. 9 7 _ {\pm 0. 1 0}
$$

$$
2.05\ _{\pm0.06}
$$

$$
4.05\ _{\pm0.11}
$$

$$
4. 6 8 _ {\pm 0. 0 9} ^ {* *}
$$

$$
2. 1 6 _ {\pm 0. 0 5} ^ {\star}
$$

$$
4. 2 1 _ {\pm 0. 0 9} ^ {\star}
$$

Table 2: Phase III human evaluation on 100 papers (3 raters/item). Values are means with 95% confidence intervals
(paired bootstrap, 1,000 resamples). *⋆p<.*05, *⋆⋆p<.*01 (Holm-corrected). Normalized Total rescales each metric
to [0,1] before summation: *F*ˆ= (*F−*1)*/*4, *C*ˆ= (*C−*1)*/*2, *E*ˆ= (*E−*1)*/*4. Engagement improvements were
statistically significant at*p<.*05.

$$
p<.05
$$

| System | UTC ↑ | Entropy ↑ | Top-1 (%) ↓ |
| --- | --- | --- | --- |
| CoT (GPT-4o) | 11 | 1.88 | 40.2 |
| Memeify | 9 | 1.74 | 45.1 |
| Self-Refine | 12 | 2.01 | 36.4 |
| SCIMEMEX (Ours) | 17 | 2.39 | 23.6 |
| w/o TSA | 13 | 2.05 | 33.8 |

Table 3: Template diversity analysis on the Phase III
evaluation set. UTC denotes the number of unique
meme templates used. Entropy measures the uniformity of the template distribution. Top-1 indicates the
proportion of memes using the most frequent template.

$$
-0.35
$$

Agent (w/o TSA) leads to reduced diversity. We
discuss this analysis in detail in AppendixK.

| Variant | Fid. | Cla. | Eng. |  | Total∆ |
| --- | --- | --- | --- | --- | --- |
| Full | 4.66 | 2.16 | 4.12 | 10.94 | – |
| w/o Contrast. | 4.55 | 2.09 | 3.95 | 10.59 | − 0 . 35 |
| w/o Feedback | 4.25 | 1.92 | 3.72 | 9.89 | ∗∗ − 1 . 05 |
| w/o TSA | 4.40 | 1.97 | 3.81 | 10.18 | ∗ − 0 . 76 |
| w/o Concisio | 4.50 | 2.02 | 3.98 | 10.50 | − 0 . 44 |
| Abstract-only | 4.32 | 1.95 | 3.88 | 10.15 | ∗ − 0 . 79 |

$$
-1.05^{\overset{\frown}{**}}
$$

$$
-0.76^{*}
$$

$$
-0.44
$$

$$
-0.79^{*}
$$

Table 4: Ablations forSCIMEMEX. Fid./Cla./Eng. denote Fidelity, Clarity, and Engagement. ∆ is relative to
the full system. Significance vs. full: *<sup>∗</sup>p<.*05, *<sup>∗∗</sup>p<.*01
(paired bootstrap).

$$
^{*}p<.05,^{**}p<.01
$$

## 5.2.3 Ablation Study

Table4summarizes the effect of removing key
components fromSCIMEMEX.Each ablation is
evaluated independently against the full system.
The steepest performance decline occurs when
the iterative feedback loop is removed (*w/o Feed-*
*back*, *−*1*.*05), reaffirming that multi-round critique–
revise interaction is central to the framework’s effectiveness.Using only the abstract instead of the
full paper context results in the second-largest reduction (*−*0*.*79), highlighting the importance of
broader document context for accurate contribution
extraction and meme generation.Removing the
template selection agent (*w/o TSA*) also degrades
performance (*−*0*.*76); in this setting, the informed
top-*k* template selection is replaced with uniform
random sampling of*k*= 5 templates from the same
template pool *M*, while keeping all other compo-

nents unchanged.Smaller but consistent declines
are observed when removing the Concisio agent
(*w/o Concisio*, *−*0*.*44) or contrastive conditioning
(*w/o Contrast.*, *−*0*.*35), indicating that both components provide complementary improvements.

Beyond the ablation evidence, we directly evaluated Concisio’s contribution extraction on 40 randomly sampled papers. Two domain experts independently identified each paper’s key contributions
from the original manuscript, compared them with
Concisio’s generated summaries, and rated them
on 1–5 Likert scales for Coverage (whether the
main contributions were captured) and Correctness
(whether unsupported or inaccurate claims were
introduced). Concisio achieved average scores of
4.75 and 5.0 for Coverage and Correctness, respectively, indicating that its summaries preserve core
contributions while avoiding hallucinated content.
Overall, the gains ofSCIMEMEXarise from iterative feedback, broader document context, informed
template selection, and structured contrastive guidance. We provide error analysis of common failure
cases in AppendixJ.

## 6 Conclusion and Future Work

We introduce scientific meme generation, a task
that translates research ideas into accessible visual–
textual narratives. We proposeSCIMEMEX, a modular agentic framework combining insight extraction, template selection, and feedback-guided refinement. Our three-phase evaluation pipeline measures Scientific Fidelity, Clarity, and Engagement
through expert and LLM-based assessment. Experiments on 1.5k computer science papers show
thatSCIMEMEXsignificantly outperforms strong
single- and multi-agent baselines, validating the
benefits of contrastive feedback and structured coordination. Future work will extendSCIMEMEX
across domains and support HCI-driven, feedbackadaptive human–AI co-creativity with interactive
editing and constraint-guided control.

---

## Limitations

We discuss the limitations of this work below:

• **Domain Scope**: Our experiments primarily focus on computer science papers (NLP/ML/AI).
While this provides a controlled evaluation
setting, extendingSCIMEMEXto other disciplines such as biology or the social sciences may require domain-specific adaptation,
which we leave for future work.

• **Challenges with Highly Technical Content**:
Papers with dense mathematical formulations
or abstract theories are often harder to translate into clear and humorous narratives. Such
memes may assume expert-level familiarity
with technical terminology. Incorporating audience modeling or adaptive simplification
mechanisms could further democratize scientific communication⁶.

• **Subjectivity of Humor and Engagement**:
Humor, creativity, and engagement naturally
vary across audiences and cultures. Although
our three-phase evaluation pipeline mitigates
this through expert calibration and LLM-asjudge validation, broader community-level
studies could further enhance robustness and
generalizability.

• **Concisio Validation Scope:** Although Concisio achieved high expert ratings in our 40-
paper validation, this study used a small controlled sample and may reflect annotator calibration or reference-selection effects. Broader
blinded validation across domains remains future work.

• **Computational Overhead**: Although our
multi-agent framework substantially improves
performance over single-agent baselines, implementing such systems inherently incurs
higher computational and financial costs. Similar to other LLM-as-a-judge approaches, the
processing efficiency and overall expense depend on the choice of underlying models.
While our structured feedback and contrastive
refinement mechanisms mitigate redundancy,

achieving full cost efficiency and standardization across different LLM settings remains an
open challenge.

• **Dynamic Template Adaptation:** An interesting extension is to adapt meme templates
dynamically based on critic feedback. While
promising, this would require re-aligning
critic objectives with changing visual structures, as critique scores are conditioned on
a fixed visual framing. Preliminary exploration suggested that switching templates midrefinement can destabilize feedback and hinder convergence. We therefore leave dynamic
template adaptation to future work.

• **Training-Free Design:** SCIMEMEXdoes
not train task-specific model parameters and
does not rely on supervised scientific-meme
training data. While this makes the framework reproducible and applicable to a new
low-resource task, future work could learn
template-selection policies, reward models, or
critique modules as larger datasets become
available.

## Ethics

We explicitly designed our system to avoid generating offensive or disparaging content by incorporating safety critics and enforcing a hard rule
whereby such outputs automatically receive the
lowest engagement score. Nevertheless, humor is
inherently subjective, and unintended biases may
still emerge. We compensated annotators fairly in
line with academic standards, and the study was
reviewed under institutional policies and deemed
exempt from IRB approval as it posed minimal risk
and involved professional adult participants. We
emphasize thatSCIMEMEXis intended as a tool
for broadening science communication and not for
trivializing research or spreading misinformation.
We used ChatGPT only for language polishing and
proofreading the paper.

Copyright and Fair Use of Meme Imagery: The
illustrative memes shown in this paper (Figures 1,
4–7) are based on widely circulated meme templates derived from copyrighted media. Their reproduction here is non-commercial, transformative,
and used solely for academic illustration of system behavior, consistent with fair use provisions

<sup>6</sup>While our current framework targets domain-literate audiences (as noted in Section1), incorporating adaptive simplification could help bridge the gap to the general lay public.

---

for scholarly commentary. The framework generates text captions paired with externally rendered
visuals via ImageFlip or a multimodal LLM, each
of which is governed by its own terms of service.
Downstream users are responsible for ensuring
jurisdiction-appropriate use, particularly in commercial settings, and the framework is fully compatible with copyright-cleared or openly licensed
template sets for users with strict IP requirements.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama
Ahmad, Ilge Akkaya, Florencia Leoni Aleman,
Diogo Almeida, Janko Altenschmidt, Sam Altman,
Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report.*arXiv preprint arXiv:2303.08774*.
Anthropic. 2023.Claude: An ai assistant by anthropic.
Model card and technical overview.
Ron Artstein and Massimo Poesio. 2008a.Inter-coder
agreement for computational linguistics.*Computa-*
*tional Linguistics*, 34(4):555–596.
Ron Artstein and Massimo Poesio. 2008b.Inter-coder
agreement for computational linguistics.*Comput.*
*Linguistics*, 34(4):555–596.
Holly M Bik and Miriam C Goldstein. 2013. An introduction to social media for scientists.*PLoS biology*,
11(4):e1001535.
Susan Blackmore. 1999.*The meme machine*. Oxford
University Press.
Yaqi Cai, Shancheng Fang, Yadong Qu, Xiaorui Wang,
Meng Shao, and Hongtao Xie. 2025.Itermeme:
Expert-guided multimodal llm for interactive meme
creation with layout-aware generation. In*Proceed-*
*ings of the Thirty-Fourth International Joint Con-*
*ference on Artificial Intelligence, IJCAI-25*, pages
720–728. International Joint Conferences on Artificial Intelligence Organization. Main Track.
Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu,
Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan
Liu. 2024.Chateval: Towards better llm-based evaluators through multi-agent debate. In*The Twelfth*
*International Conference on Learning Representa-*
*tions, ICLR 2024, Vienna, Austria, May 7-11, 2024*.
OpenReview.net.
Yuyan Chen, Songzhou Yan, Zhihong Zhu, Zhixu Li,
and Yanghua Xiao. 2024. Xmecap: Meme caption
generation with sub-image adaptability. In*Proceed-*
*ings of the 32nd ACM International Conference on*
*Multimedia*, pages 3352–3361.
Richard Dawkins. 1976.*The selfish gene*. Oxford University Press.

DeepMind. 2023.Gemini: A family of highly capable
multimodal models. Google DeepMind Technical
Report.
GROBID Developers. 2008.Grobid. https:
//github.com/grobidOrg/grobid.*Preprint*,
swh:1:dir:dab86b296e3c3216e2241968f0d63b68e8209d3c.
Yue Dong, Andrei Romascanu, and Jackie Chi Kit Cheung. 2021.Discourse-aware unsupervised summarization for long scientific documents. In*Proceed-*
*ings of the 16th Conference of the European Chap-*
*ter of the Association for Computational Linguistics:*
*Main Volume, EACL 2021, Online, April 19 - 23,*
*2021*, pages 1089–1102. Association for Computational Linguistics.
Baban Gain, Saswati Dana, Udit Sharma, Arnab Kumar Mondal, Prathosh AP, Dinesh Garg, and Amith
Singhee. 2026.Task-aware model merging via fisherweighted median.
Marek Górka. 2014. The meme as an example of
carnivalized internet communication.*Kwartalnik*
*Naukowy OAP UW" e-Politikon"*, pages 215–242.
J. P. Guilford. 1967.*The nature of human intelligence*.
McGraw-Hill.
Karl Moritz Hermann, Tomás Kociský, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman,
and Phil Blunsom. 2015.Teaching machines to read
and comprehend. In*NIPS*, pages 1693–1701.
Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and
Yejin Choi. 2020.The curious case of neural text
degeneration. In*8th International Conference on*
*Learning Representations, ICLR 2020, Addis Ababa,*
*Ethiopia, April 26-30, 2020*. OpenReview.net.
EunJeong Hwang and Vered Shwartz. 2023.MemeCap:
A dataset for captioning and interpreting memes.
In*Proceedings of the 2023 Conference on Empir-*
*ical Methods in Natural Language Processing*, pages
1433–1445, Singapore. Association for Computational Linguistics.
Kirstynn Joseph. 2023.Add memes to your scicomm
toolbox. Fancy Comma.
Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. 1996. Reinforcement learning: A
survey.*Journal of artificial intelligence research*,
4:237–285.
Daniel Kahneman and Amos Tversky. 1984. Choices,
values, and frames.*American psychologist*,
39(4):341.
Takuro Kawada, Shunsuke Kitada, Sota Nemoto, and Hitoshi Iyatomi. 2025. Sciga: A comprehensive dataset
for designing graphical abstracts in academic papers.
*arXiv preprint arXiv:2507.02212*.
Doyoung Kim, Jaehyeok Doo, and Minjoon Seo. 2026.
TSLM: tree-structured language modeling for divergent thinking.*CoRR*, abs/2601.22688.

---

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners.*Advances in*
*neural information processing systems*, 35:22199–
22213.
Klaus Krippendorff. 2018.*Content Analysis: An In-*
*troduction to Its Methodology*, 4th edition. SAGE
Publications.
Sandeep Kumar, Guneet Singh Kohli, Tirthankar
Ghosal, and Asif Ekbal. 2024a.Longform multimodal lay summarization of scientific papers: Towards automatically generating science blogs from
research articles. In*Proceedings of the 2024 Joint*
*International Conference on Computational Linguis-*
*tics, Language Resources and Evaluation (LREC-*
*COLING 2024)*, pages 10790–10801, Torino, Italia.
ELRA and ICCL.
Sandeep Kumar, Mohit Sahu, Vardhan Gacche,
Tirthankar Ghosal, and Asif Ekbal. 2024b.‘quis
custodiet ipsos custodes?’ who will watch the watchmen? on detecting AI-generated peer-reviews. In
*Proceedings of the 2024 Conference on Empirical*
*Methods in Natural Language Processing*, pages
22663–22679, Miami, Florida, USA. Association
for Computational Linguistics.
Susanne Landis and Tom Duscher. 2022. Visual science
communication: the next generation scientific poster.
*Qeios ID*, page 8OOHNS.
Alan I Leshner. 2003. Public engagement with science.
Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang,
Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and
Zhaopeng Tu. 2024.Encouraging divergent thinking
in large language models through multi-agent debate.
In*Proceedings of the 2024 Conference on Empiri-*
*cal Methods in Natural Language Processing*, pages
17889–17904, Miami, Florida, USA. Association for
Computational Linguistics.
Chin-Yew Lin. 2004. Rouge: A package for automatic
evaluation of summaries. In*Text Summarization*
*Branches Out: Proceedings of the ACL-04 Workshop*,
pages 74–81.
Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler
Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon,
Nouha Dziri, Shrimai Prabhumoye, Yiming Yang,
and 1 others. 2023. Self-refine: Iterative refinement
with self-feedback.*Advances in Neural Information*
*Processing Systems*, 36:46534–46594.
Esther Marín-González, Davide Malmusi, Lluís Camprubí, and Carme Borrell. 2017. The role of dissemination as a fundamental part of a research project:
lessons learned from sophie.*International journal of*
*health services*, 47(2):258–276.
Rachel Myers. 2022.Function of memes in adolescent communication: A theoretical review.*Seeds of*
*Science*.

Sandra Oska, Edgar Lerma, and Joel Topf. 2020. A
picture is worth a thousand views: a triple crossover
trial of visual abstracts to examine their impact on
research dissemination.*Journal of Medical Internet*
*Research*, 22(12):e22327.
Allan Paivio. 1990.*Mental representations: A dual*
*coding approach*. Oxford university press.
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-
Jing Zhu. 2002.Bleu: a method for automatic evaluation of machine translation. In*Proceedings of the*
*40th Annual Meeting of the Association for Compu-*
*tational Linguistics (ACL)*, pages 311–318.
Abel L Peirson V and E Meltem Tolunay. 2018. Dank
learning: Generating memes using deep neural networks.*arXiv preprint arXiv:1806.04510*.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021a. Learning transferable visual models from natural language
supervision. In*Proceedings of the 38th International*
*Conference on Machine Learning (ICML)*, pages
8748–8763.
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya
Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark,
Gretchen Krueger, and Ilya Sutskever. 2021b.Learning transferable visual models from natural language
supervision. In*International Conference on Machine*
*Learning*.
Matthew Renze and Erhan Guven. 2024. Self-reflection
in llm agents: Effects on problem-solving performance.*arXiv preprint arXiv:2405.06682*.
Diana K Riser, Stephanie D Clarke, and Allison N Stallworth. 2020. Scientific memes: Using the language
of social media to improve scientific literacy and
communication in lifespan development.*Psychology*
*Learning & Teaching*, 19(3):275–289.
R. Keith Sawyer. 2012.*Explaining creativity: The sci-*
*ence of human innovation*. Oxford University Press.
Athar Sefid and C. Lee Giles. 2022.Scibertsum: Extractive summarization for scientific documents. In
*Document Analysis Systems - 15th IAPR Interna-*
*tional Workshop, DAS 2022, La Rochelle, France,*
*May 22-25, 2022, Proceedings*, volume 13237 of
*Lecture Notes in Computer Science*, pages 688–701.
Springer.
Athar Sefid, Prasenjit Mitra, and Lee Giles. 2021. Slidegen: an abstractive section-based slide generator for
scholarly documents. In*Proceedings of the 21st*
*ACM Symposium on Document Engineering*, pages
1–4.
Claude E. Shannon. 1948.A mathematical theory of
communication.*Bell Syst. Tech. J.*, 27(3):379–423.

---

Grishma Sharma, Deepak H. Sharma, and M. Sasikumar. 2024.Summarizing long scientific documents
through hierarchical structure extraction.*Nat. Lang.*
*Process. J.*, 8:100080.
Katelyn Shibilski. 2021.*“Meme, myself, and I:” Self-*
*directed effects in meme-centered pedagogy*. Ph.D.
thesis, Florida Southern College.
Limor Shifman. 2013.*Memes in digital culture*. MIT
press.
Tao Sun, Enhao Pan, Zhengkai Yang, Kaixin Sui, Jiajun
Shi, Xianfu Cheng, Tongliang Li, Wenhao Huang,
Ge Zhang, Jian Yang, and Zhoujun Li. 2025.P2P: automated paper-to-poster generation and fine-grained
benchmark.*CoRR*, abs/2505.17104.
Richard S Sutton and Andrew G Barto. 2018.*Reinforce-*
*ment learning: An introduction*. MIT press.
Zusheng Tan, Xinyi Zhong, Jing-Yu Ji, Wei Jiang, and
Billy Chiu. 2025. Enhancing large language models
for scientific multimodal summarization with multimodal output. In*Proceedings of the 31st Inter-*
*national Conference on Computational Linguistics:*
*Industry Track*, pages 263–275.
Shohei Tanaka, Hao Wang, and Yoshitaka Ushiku. 2024.
Scipostlayout: A dataset for layout analysis and layout generation of scientific posters.*arXiv preprint*
*arXiv:2407.19787*.
E. Paul Torrance. 1988.*The nature of creativity: Con-*
*temporary psychological perspectives*. Cambridge
University Press.
Laura Tran. 2024.From lab to likes: Socializing science
through humor. The Scientist.
Laura Van Eperen and Francesco M Marincola. 2011.
How scientists use social media to communicate
their research.*Journal of Translational Medicine*,
9(1):199.
Suryatej Reddy Vyalla and Vishaal Udandarao. 2020.
Memeify: A large-scale meme generation system. In
*CoDS-COMAD 2020: 7th ACM IKDD CoDS and*
*25th COMAD, Hyderabad India, January 5-7, 2020*,
pages 307–311. ACM.
Suryatej Reddy Vyalla, Vishaal Udandarao, and Tanmoy Chakraborty. 2019.Memeify: A large-scale
meme generation system.*Proceedings of the 7th*
*ACM IKDD CoDS and 25th COMAD*.
Chi Wang, Qingyun Wu, and the AG2 Community.
2024a.Ag2: Open-source agentos for ai agents.
Available at https://docs.ag2.ai/.
Han Wang and Roy Ka-Wei Lee. 2024a.Memecraft:
Contextual and stance-driven multimodal meme generation. In*Proceedings of the ACM on Web Confer-*
*ence 2024, WWW 2024, Singapore, May 13-17, 2024*,
pages 4642–4652. ACM.

Han Wang and Roy Ka-Wei Lee. 2024b. Memecraft:
Contextual and stance-driven multimodal meme generation. In*Proceedings of the ACM Web Conference*
*2024*, pages 4642–4652.
Hao Wang, Shohei Tanaka, and Yoshitaka Ushiku.
2024b. Scipostlayout: A dataset for layout analysis
and layout generation of scientific posters. In*Pro-*
*ceedings of the IEEE/CVF Conference on Computer*
*Vision and Pattern Recognition*, pages 8136–8141.
B Warren. 2003. Current challenges and choices in
scientific publication baylor university medical centre
proceedings, v 16(4).*Retrieved Nov*, 27:2008.
Bradley E Wiggins and G Bret Bowers. 2015. Memes as
genre: A structurational analysis of the memescape.
*New media & society*, 17(11):1886–1906.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
Tom Griffiths, Yuan Cao, and Karthik Narasimhan.
2023.Tree of thoughts: Deliberate problem solving
with large language models. In*Advances in Neural*
*Information Processing Systems 36: Annual Confer-*
*ence on Neural Information Processing Systems 2023,*
*NeurIPS 2023, New Orleans, LA, USA, December 10*
*- 16, 2023*.

## A Implementation Details

We implementSCIMEMEXin Python using the autogen framework (Wang et al.,
2024a). Experiments were conducted with
the following models: gpt-4o-2024-08-06
(GPT-4o), anthropic.claude-3.5-sonnet
(Claude-3.5), gemini-1.5-pro (Gemini), and
llama-3.1-70b-instruct (Llama). For all models, we set the temperature to 0 and the random
seed to 1111 to ensure reproducibility. At the
time of experimentation, API usage costs were as
follows: GPT-4o—$2.50 per million input tokens
and $10.00 per million output tokens; Claude-3.5
Sonnet—$3.00 (input) and $15.00 (output) per
million tokens; and Gemini-1.5 Pro—$1.25
(input) and $5.00 (output) per million tokens. For
automated evaluation, we constructed a dataset of
2k papers randomly sampled from the Computer
Science (AI/NLP/ML) categories on arXiv⁷,
covering the years 2022–2023. Since no training
was required, we used 1.5k papers and partitioned
them into validation (30%) and test (70%) splits.
The remaining papers were reserved for Phase II
and Phase III human evaluations, discussed later in
the paper. We set the exploration parameter *k*= 5
for all experiments. We set *α*, *β*, and *γ* to 1 for this
experiment; that is, all three evaluation metrics are
given equal importance. However, users can adjust

$$
\alpha,\beta,
$$

$$
\gamma
$$

7
https://arxiv.org/ these values according to their preferences.We
evaluate all baselines using the same task-specific
inputs: MAD/Self-Refine/ChatEval receive the
same prompt and the same abstract + Concisio
contrastive summary, and Memeify/MemeCraft
are given the same Concisio summary, template
list, and rubric constraints (with only minimal
formatting changes to match their single-pass
interfaces).All agents inSCIMEMEXand the
baseline frameworks were instantiated using the
same underlying LLM for a fair comparison.
We mitigate potential self-preference bias in
LLM-as-a-judge evaluation by using a stronger
judge model (GPT-5.1) rather than the GPT-4o
backbone used in generation, and by validating
results across heterogeneous backbone models
and human evaluation. We preprocess research
paper PDFs using GROBID (Developers,2008)
to extract structured document sections and plain
text. The extracted representation includes the title,
abstract, and main body sections, while references,
figures, tables, and appendices are excluded since
they are not required for the meme-generation
pipeline.We preprocess research paper PDFs using
GROBID (Developers,2008) to extract structured
document sections and plain text. The extracted
representation includes the title, abstract, and main
body sections, while references, figures, tables,
and appendices are excluded since they are not
required for the meme-generation pipeline.Prompt
for each agents are discussed in the Appendix
below. We use two nodes of NVIDIA A100 GPUs
for our experiments.

## B Baselines

We compare our multi-agent approach against several baselines, including**CoT**(Kojima et al.,2022),
**Self-Reflection**(Renze and Guven,2024),**Self-**
**Refine**(Madaan et al.,2023),**MAD**(Liang et al.,
2024), and**ChatEval**(Chan et al.,2024) (see AppendixDfor details). We adapt established memegeneration systems such as Memeify (Vyalla et al.,
2019) and MemeCraft (Wang and Lee,2024b) by
using their template-conditioned caption generation modules, which operate solely on textual inputs. Since our task does not involve image inputs,
other multimodal meme-generation methods (e.g.,
Dank Learning (Peirson V and Tolunay,2018),
XMeCap (Chen et al.,2024), CLIP-based models (Radford et al.,2021b;Hwang and Shwartz,
2023)) cannot be used as baselines because they

fundamentally rely on image embeddings or visionlanguage components that are incompatible with
our text-only setting.

## C Evaluation Metrics

The details the proposed evaluations is below:-

• **Scientific Fidelity**We introduce the**Fidelity**
**Score**, which quantifies the extent to which
a meme faithfully preserves the*core scien-*
*tific claim*of the underlying research paper.
Following prior work on LLM-as-judge evaluation, the model is instructed to: (i) extract the
central contribution of the paper, (ii) identify
the claim implied by the meme, and (iii) compare the two along three axes: factual correctness, preservation of scope/limitations, and
integrity of meaning. The evaluator returns
both a**discrete score (1–5)**and a structured
reasoning trace, where 5 denotes exact fidelity
and 1 indicates complete scientific distortion.

• **Clarity and Interpretability**We measure
how well a meme communicates the intended scientific message to its target audience (graduate-level NLP/ML researchers).
To this end, we adopt the**FRI (First-Read In-**
**terpretability) Score**, a three-point rubric inspired by interpretability and comprehension
evaluation protocols. The evaluator first interprets the meme without access to the groundtruth claim, then compares this interpretation
against the intended message, and finally assigns a score: **3 = clear**,**2 = partially clear**,
**1 = unclear**. This process explicitly disentangles the meme’s communicative clarity from
its scientific accuracy.

• **Engagement Potential**Finally, we evaluate
the meme’s likelihood of resonating with and
being shared by the intended scientific community. The**Engagement Potential Score**(1–
5) captures predicted resonance along three
factors: humor strength, synergy between
caption and template, and non-offensiveness.
Importantly, factual accuracy and clarity are
treated as orthogonal and excluded from this
metric. A*hard rule*is enforced whereby
any meme containing disparaging or offensive
content automatically receives a score of 1,
ensuring that potential virality is not rewarded
at the expense of community norms.

---

## D Baseline Details

• **CoT**(Kojima et al.,2022): This approach
concatenates a trigger sentence “Let’s think
step by step” to the task. Our CoT baseline is
implemented as a strong rubric-guided singleagent prompt rather than a naïve one-shot instruction. The prompt was selected through
iterative prompt engineering on a held-out validation set and explicitly guides the model to
reason about prior limitations, summarize the
paper’s core contribution, select an appropriate meme template, and generate a caption
conditioned on the evaluation rubric.

• **Self-Reflection**(Renze and Guven,2024): Incorporates a post-hoc reflection step where
the LLM reviews its own mistakes (using the
correct answer as feedback), generates reflections (e.g., advice, explanations, or alternative
solutions), and then re-answers the question.

• **Self-Refine**(Madaan et al.,2023): Self-
Refine is an approach that iteratively improves
LLM outputs by generating an initial response,
then using the same LLM to provide feedback
and refine its output through repeated selfevaluation.

• **Multi-Agent Debate (MAD)**: (Liang et al.,
2024) Engages multiple agents in a “tit-fortat” debate, with a judge managing the process
to reach a final solution. This framework encourages divergent thinking in LLMs, which
is beneficial for tasks requiring deep contemplation.

• **ChatEval**: (Chan et al.,2024) In the In the
best-performing One-By-One approach, debater agents take turns responding in a fixed
order, with each response appended to the chat
history so that subsequent agents explicitly
condition on all prior contributions.

## D.1 Phase II: LLM-as-Judge Validation

## (Prompts, Calibration, and Reliability)

Having formalized the rubrics, we implemented
them as structured prompts and evaluated their reproducibility under an LLM-as-judge paradigm
(Cai et al.,2025) using GPT-4.1 (Achiam et al.,
2023) on a new sample of 300 memes. Six independent domain experts (senior PhD students and
early-career researchers, not involved in Phase I)
annotated each meme on all three dimensions, with

three raters per item. The curated set of memes
from Phase I served as the training corpus for all
annotators. Annotators underwent two rounds of
structured training. In the first round, they independently annotated 30 memes drawn from the
calibration set, followed by a group discussion reviewing common errors and divergent interpretations. Feedback from this session was used to clarify borderline cases, particularly concerning the
separation between*Fidelity*and*Clarity*. In the second round, annotators re-annotated a fresh set of
30 memes balanced across difficulty levels. We
employed an iterative feedback process to continuously monitor annotation quality and resolve ambiguities. When inconsistencies or conflicts arose,
an expert reviewed disputed cases and provided
clarifications in subsequent rounds. Periodic crosschecks ensured stable interpretation of the scoring
criteria. The prompts used by LLM as judge is
discussed inI.

Inter-annotation agreement was measured using
standard reliability metrics (Artstein and Poesio,
2008a;Krippendorff,2018), yielding substantial
human–human consistency (*τ*= 0*.*69, *κ*= 0*.*66,
*r*= 0*.*72, *α*= 0*.*68). GPT-4.1, prompted with
the same rubric under controlled settings, was
then applied to the identical dataset. Its ratings
showed high concordance with human judgments
(*τ*= 0*.*71, *κ*= 0*.*68, *r*= 0*.*74). These values fall
within or above commonly accepted thresholds for
strong inter-annotator reliability in computational
linguistics (Artstein and Poesio,2008b;Krippendorff,2018). Importantly, the LLM-as-judge framework achieved higher internal consistency than additional human raters with limited domain expertise, reinforcing its validity as a scalable proxy.

## E Phase II Evaluation Details

**Dataset Construction.**The Phase II dataset comprised 300 memes, drawn to balance (i) paper topics (NLP, vision-language, multimodal), (ii) meme
templates (commonly used formats vs. niche formats), and (iii) generation modes (human-authored
vs. LLM-generated). The set was distinct from the
80-paper sample used in Phase I to avoid circularity
between rubric design and validation.

**Annotators.**We recruited twelve domain experts
(senior PhD students and early-career researchers
in NLP/ML) who had not participated in Phase I.
Each meme was annotated by three independent
raters across the three dimensions (Fidelity, Clarity

---

& Interpretability, Engagement Potential), yielding
900 judgments per dimension. To calibrate interpretations, raters completed a 20-meme practice round
with group discussion. Annotation was performed
independently thereafter.

**Inter-Annotator Agreement (IAA).**We report
four complementary measures:-**Weighted Co-**
**hen’s** *κ* (quadratic weights) for ordinal consistency.
-**Kendall’s** *τ* for rank-order similarity.-**Pear-**
**son’s** *r* between leave-one-rater-out scores and
item means, capturing correlation on continuous
scale use.-**Krippendorff’s** *α* (ordinal distance
metric) as a robustness check (Artstein and Poesio,
2008a;Krippendorff,2018).

$$
\tau
$$

For *κ* and *τ*, we compute pairwise values across
all rater pairs per item and then macro-average
across items. For *r*, each item’s mean rating was
compared to the left-out rater in rotation and averaged. For *α*, we aggregated across all raters per
item using ordinal distance *|i−j|*. Confidence
intervals (95%) were computed via nonparametric
bootstrap with 1,000 item-level resamples.

$$
r,
$$

$$
\alpha,
$$

$$
|i-j|
$$

**Results.**Human reliability was substantial across
dimensions, with averages of Kendall’s *τ*= 0*.*69,
weighted *κ*= 0*.*66, Pearson’s *r*= 0*.*72, and Krippendorff’s *α*= 0*.*68. These fall within or above
established thresholds for strong reliability in computational linguistics. Importantly, we report raw
agreement prior to adjudication, consistent with
best practices (Artstein and Poesio,2008a).

$$
\tau=0.69
$$

$$
\kappa=0.66
$$

$$
r=0.72
$$

$$
\alpha=0.68
$$

**LLM-as-Judge Setup.**GPT-4o was instantiated
with temperature *T*= 0 and fixed system role instructions to minimize variance. The model was
prompted with the finalized rubric for each dimension, producing structured outputs with both justification and numeric ratings. To align with human
scales, outputs were discretized to the nearest Likert category (1–5 for Fidelity/Engagement, 1–3 for
Clarity). Agreement between GPT-4o and human
means was then computed using the same metrics
as above.

$$
T=0
$$

**LLM vs. Human Concordance.**GPT-4o
achieved high agreement with expert judgments:
Kendall’s *τ*= 0*.*71, *κ*= 0*.*68, and *r*= 0*.*74,
indicating strong alignment and exceeding the
reliability achieved by a secondary pool of
non-expert annotators. This supports the validity
of the LLM-as-judge paradigm as a scalable proxy
when expert annotation is costly.

$$
\tau=0.71
$$

$$
\kappa\;=\;0.68
$$

$$
r\;=\;0.74
$$

## E.1 Annotators’ Pay

All annotators were compensated at rates consistent with fair academic practice for expert annotation. Annotators received $30 USD per hour,
benchmarked against prevailing rates for graduatelevel research assistance in NLP/ML. Pilot timings
indicated that annotating 20 memes required approximately one hour, yielding an effective permeme rate of $1.50. Compensation was prorated
for partial batches to ensure fairness, and annotators were paid regardless of whether their annotations were ultimately retained. Annotators in
Phases II and III were recruited independently of
the author team, participated voluntarily, and provided informed consent. They were free to withdraw at any time without penalty. No annotator
was paid below local minimum wage, and compensation was standardized across geographic regions
to avoid disparities. Total expenditure for the study
was approximately $4,500 USD. This study was
reviewed under our institution’s human subjects
research policies and deemed exempt from formal
IRB approval, as it involved minimal risk and professional adult participants.

## F Annotation Guidelines (Shared with Annotators).

Please follow the steps below carefully when evaluating each meme. You will assign three scores—
**Scientific Fidelity**,**Clarity**, and**Engagement**—
according to the provided rubrics. Read only the
materials shared here; do not access the full paper.
For each item, the materials shared with annotators consisted of the generated meme, the evaluation rubric, the paper title, the abstract, contribution statements from the introduction, and a short
related-work context excerpt. Thus, the “short excerpt” referenced below was not the full paper, but
a controlled reference constructed from the paper
abstract, introduction-level contribution statements,
and related-work context.

1. **Initial Reading:** Look at the meme independently without referring to any additional material. Note your immediate understanding of
what scientific idea it conveys.

2. **Clarity and Engagement:** Based only on the
meme’s text and visuals, rate how clearly it
communicates a scientific concept (*Clarity*) and
how effectively it captures attention or evokes interest among a technical audience (*Engage-*
*ment*).

3. **Reference Review:** After completing Step 2,
read the short excerpt provided to you. This
excerpt summarizes the paper’s main claim or
contribution and may come from the abstract
or another section. Do**not**read the full paper.

4. **Fidelity:** Compare your initial interpretation
of the meme with the reference excerpt and
rate how accurately the meme preserves the
intended scientific meaning.

5. **Final Check:** Review all your scores for consistency and neutrality. Avoid personal or stylistic bias, and assign the lowest engagement
score (1) to any meme containing offensive,
disrespectful, or disparaging content.

Your judgments should reflect how effectively
each meme communicates the intended scientific
idea to a graduate-level technical audience, maintaining both objectivity and respectfulness throughout.

## G Human Evaluation Score Distributions

## important G Human Evaluation Score Distributions

To examine whether human agreement was driven
by annotators clustering around midpoint Likert
scores, we report the Phase III score distributions
forSCIMEMEX. As shown in Table5, ratings are
not concentrated around the midpoint. For Fidelity
and Engagement, the modal rating is 5, while only
8% and 16% of ratings fall at score 3, respectively.
Clarity, which uses a 1–3 scale, also shows variation across its available categories.

| Dimension | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- |
| Fidelity (1–5) | 1% | 2% | 8% | 10% | 79% |
| Engagement (1–5) | 1% | 5% | 16% | 28% | 50% |
| Clarity (1–3) | 11% | 62% | 27% | – | – |

Table 5:Phase III human evaluation score distributions
forSCIMEMEX. The distributions show that ratings are
not concentrated around midpoint scores.

## H Agent Prompts

We list below the exact system prompts used for
each agent in our framework. All agents run
in human_input_mode=NEVER and interact solely
through textual exchanges.

## Concisio

You are a comparative research analyst.
Your task is to deeply analyze a research
paper to extract the main conceptual and
technical differences between past work and
this paper’s contributions.

Step 1: First write the below information
below :- - Prior Work (Old Ideas): Summarize how this problem was tackled previously, including model names, strategies,
or limitations. - Proposed Approach (New
Idea): What method, model, or strategy is
introduced by this paper?- Core Differences: Explain what changes in methodology, architecture, or objective, why it matters, and whether tradeoffs or assumptions
are introduced. Be precise, technical, and
avoid generic summaries.

Step 2: Using the information, produce a
single paragraph (under 100 words) that captures the**essential difference**.

Your output should: - State what was done
before - Explain what this paper does differently - Highlight why this difference is
important

Avoid vagueness or filler. Do not invent
facts. Return only the summary paragraph.

## Generator

You are a creative meme generator specializing in academic research. Generate diverse,
creative memes that contrast prior work vs.
new contributions.

Return only the meme template and text in
a structured format.

## Meme_Selector

You are a meme strategy selector. Given a
paper’s key ideas and a list of meme templates, choose the top-*k* templates that best
highlight the contrast between prior work
and contributions.

## I LLM Prompts for Evaluation

This appendix lists the exact prompts used for
our LLM-based evaluation pipeline. We show the
systemandusermessages for each evaluator.

---

**Algorithm 1:** Agentic Exploration–Exploitation Framework for Scientific Meme Generation
**Input:** Research paper*p*, meme template set*M*, number of iterations*T*, top-*k*templates
**Output:** Final best meme*B*<sub>p</sub>with evaluation scores
//Step 1: Paper Summarization (Contrastive Insight Extraction)
Obtain a structured summary *S*<sub>p</sub>using the Concisio_Insight agent (prior vs. new contributions),
followed by theConcisioagent to generate concise reflections of key differences.
//Step 2: Initial Exploration Phase
Select top-*k*meme templates*T*<sub>p</sub>*⊆M*using theTemplate_Selectoragent on*S*<sub>p</sub>.
Generate*k*initial captioned memes*{m₁,...,m*<sub>k</sub>*}*with theGeneratoragent.
Evaluate each meme via theCritiqueagent to obtain a score*s*(*m*).
Identify the best meme*m<sup>+</sup>* and worst meme*m<sup>−</sup>* based on*s*(*m*).
//Step 3: Iterative Contrastive Exploitation
**for***t*= 1**to***T***do**
Select new candidate templates*T*<sup>pt</sup>using theTemplate_Selectoragent.
+ −
Generate contrastive memes conditioned on(*m,m*)using theContrastive_Generator.
Evaluate each new meme via theCritiqueagent to obtain score*s*(*m*).
+
**if**max*s*(*m*)*> s*(*m*)**then**
Update*m<sup>+</sup> ←m*where*m*has the highest score.
//Step 4: Final Output
+
Return*B*<sub>p</sub>=*m* along with final evaluation scores(*f,c,e*).

$$
\mathcal{B}_{p}
$$

$$
S_{p}
$$

$$
T_{p}\subseteq\mathcal{M}
$$

$$
S_{p}.
$$

$$
\{m_{1},\ldots,m_{k}\}
$$

$$
m^{+}
$$

$$
m^{-}
$$

$$
s(m)
$$

$$
T_{p}^{t}
$$

$$
\left(m^{+},m^{-}\right)
$$

$$
s(m)>s(m^{+})
$$

$$
\mathcal{B}_{p}=m^{+}
$$

$$
(f,c,e)
$$

## Fidelity Score Evaluator (System Prompt)

You are an expert scientific reviewer. Your
only task is to evaluate*scientific fidelity*of
a meme generated from a research paper.
**Definition:** Scientific fidelity = the degree
to which the meme faithfully preserves the
research paper’s**core claim or contribu-**
**tion**without distortion.
It does not measure humor, creativity, clarity, or engagement — only whether the scientific meaning remains accurate.
**Evaluation Steps:** 1. Extract Core Claim.
2. Extract Meme Claim. 3. Compare for
factual correctness, scope preservation, and
integrity of meaning.
**Scoring Rubric (1–5):**- 5 (Exact Fidelity):
Meme precisely captures the paper’s core
contribution. - 4 (High Fidelity): Captures
main claim correctly but omits nuance.-
3 (Partial Fidelity): Conveys general topic
but not specific contribution. - 2 (Low Fidelity): Distorted or misleading framing. -
1 (No Fidelity): Scientifically inaccurate or
unrelated.

**Output Format (JSON):**{
"score": [1–5], "reasoning": "...",
"correct_claim_summary": "...",
"meme_claim_summary": "...",
"scope_check": "...", "factual_status":
"...", "key_misrepresentations": ["..."] }

## Clarity & Interpretability Evaluator (User Prompt)

**Context**You are an AI assistant tasked with
evaluating the clarity and interpretability of
a meme designed to communicate a novel
scientific idea. The target audience is graduate students and early-career researchers in
CS/NLP.

**The Meme to Evaluate**- Meme Template:
{meme_template_description} - Meme Caption: {meme_caption}

**The Ground Truth (Core Scientific Mes-**
**sage)**This is the idea the meme is supposed
to communicate. Do NOT use this information for your initial interpretation. - Core
Message: {research_idea}

**Task**1. Initial interpretation based only on
meme template and caption. 2. Compare with the ground truth. 3. Assign FRI Score
(1–3). 4. Provide a brief justification.
**Scoring Rubric**- 3 (Clear): Meme effectively conveyed the core scientific message.
- 2 (Partially Clear): Meme conveyed part
of the message but left out key details or nuance. - 1 (Unclear): Meme failed to convey
the intended message.
**Output Format (JSON)**{ "fri_score": 1–3,
"justification": "..." }

## Engagement Potential Evaluator (System Prompt)

You are an expert scientific reviewer. You
will assign ONE overall **Engagement** Potential score (1–5) to a scientific meme.
**Definition (what to judge):**- Predicted resonance and shareability within the scientific
audience. - Humor strength, template synergy, and offensiveness.
**Orthogonality (what NOT to judge):**- Do
NOT judge fidelity/factuality of the meme.
- Do NOT judge clarity or pedagogical quality.
**Hard Rule:** If meme is offensive/disparaging→score = 1.
**Scoring Guide:**- 5: Exceptional resonance;
highly shareable.- 4: Strong resonance;
relatable. - 3: Moderate; niche appeal. - 2:
Low resonance; weak joke.- 1: Doesn’t
land OR offensive.
**Output Format (JSON):**{ "score": <1–5>,
"reasoning": "*≤*20 words" }

## J Error Analysis

**When Brevity Hurts Fidelity.**Because memes
allow only a few words, papers with multiple contributions often get reduced to just one aspect. This
leads to low fidelity: the meme captures part of the
work but ignores other core ideas.*Example.*A paper proposing both a*new multilingual dataset*and
a*novel evaluation metric*was compressed into a
meme (Drake template) contrasting “Old baselines”
vs. “Our new dataset”. The evaluation contribution
was entirely omitted, reducing the fidelity of the
meme to only half of the actual paper.

**Challenges in Abstracting Highly Technical**
**Content.**For mathematically heavy or technical papers, the model often struggles to find the

right abstraction. It either retains jargon verbatim
or collapses it into vague placeholders. Both extremes reduce clarity and humor.*Example.*A paper
on*trust-region Fisher merging for LLaMA models*
was turned into a Two Buttons meme with captions
“complicated math stuff” vs. “more training data”.
The key novelty—trust-region optimization—was
lost, making the meme inaccurate and uninformative.

**Lack of General Readability.**Even when fidelity is preserved, memes often assume substantial
domain knowledge. This makes them unclear to
readers outside the specific subfield. For example, a
meme referencing “BLEU vs. COMET—pick your
fighter” accurately captured evaluation debates in
MT research but would confuse audiences unfamiliar with these metrics, reducing accessibility and
communicative reach.

## K Additional Analysis: Template Diversity

We provide additional details on the template diversity analysis. Automatic meme generation models
are prone to*template collapse*, where a small subset of popular formats dominates outputs, reducing
expressive variety. We analyze the empirical distribution of meme templates selected by each system
on the Phase III evaluation set using Unique Template Count (UTC), Shannon entropy, and Top-1
template frequency.

As shown in Table3,SCIMEMEXexhibits substantially higher template diversity than singleagent prompting and prior multi-agent baselines,
with higher entropy and lower template dominance.
Removing the Template Selector Agent (w/o TSA)
reduces diversity, confirming that gains stem from
the structured, contrast-aware template selection
step applied once. Notably, increased template
diversity does not degrade scientific quality, remaining compatible with improvements in fidelity,
clarity, and engagement.

**Manual TSA Failure Audit.**Since the Template
Selector Agent always returns a top-*k* subset from
the fixed template library *M*, we manually inspected whether the selected templates were appropriate rather than whether the agent returned
no template. In a spot check of 25 randomly sampled papers, we judged the top-*k* selected templates
for whether at least one template was usable for
the paper’s contrastive framing. We observed only one case in which all *k* selected templates were
unusable, suggesting that complete TSA selection
failure was rare in this audit.

## L Examples of a Few Generated Memes from SciMemeX

Predicting tokens sequentially and just praying the first try is the right answer

Calling the LLM 100 separate times for Tree-of-Thought because it can't explore on its own

Training the model to literally spit out `[SEP]` and `[FAIL]` tokens to build the search tree natively

Hitting 100% on Game of 24 and destroying ToT in inference time with just ONE coherent forward pass

Figure 3: Example of SciMemeX generated meme of
TSLM paper (Kim et al.,2026)

Reviewer at 11:59 PM:
“ChatGPT, add strengths,
weaknesses, and
2 questions 😂”
Our detector:
“Wait... that's illegal.

Figure 4: Example of a SciMemeX-generated meme
illustrating the problem of AI-generated peer reviews
from paper (Kumar et al.,2024b)

Figure4conceptually highlights the motivation
behind detecting AI-generated peer reviews. The
upper panel reflects a common real-world scenario
in which a reviewer, under tight conference deadlines, may rely on large language models such as
ChatGPT to generate a review. The lower panel
illustrates the role of our proposed detection framework, which flags such AI-generated content. This
conceptual illustration emphasizes the need for automated tools to assist editors in safeguarding the
integrity of scientific peer review.

Figure3conceptually outlines the paradigm shift
introduced by Tree-Structured Language Modeling (TSLM) (Kim et al.,2026) compared to prior
baselines. The upper panels represent traditional
approaches: standard sequential decoding, which

linearly commits to a single path, and external scaffolding methods like Tree-of-Thought (ToT) (Yao
et al.,2023), which require computationally expensive, redundant sampling. The lower panels
illustrate the conceptual leap of TSLM. By training
the model to natively emit structural tokens ([SEP]
and [FAIL]), the model internalizes the search procedure. The final "galaxy brain" tier encapsulates
the ultimate result: achieving 100% accuracy on
complex tasks like the Game of 24 in a single, coherent forward pass, drastically outperforming the
efficiency and scalability of external search methods.

Interference-aware merging (TIES)

Sensitivity-aware merging (Fisher merging)

DRIFT-MEDIAN

Fine... let's work together.💕

Figure 5: Example of SciMemeX generated meme of
DRIFT-MEDIAN paper (Gain et al.,2026)

Figure5conceptually illustrates the key insight
behind DRIFT-MEDIAN compared to prior model
merging approaches. The left and right panels represent two dominant paradigms in existing literature: interference-aware methods such as TIES,
which resolve sign conflicts but ignore parameter
importance, and sensitivity-aware approaches such
as Fisher merging, which account for parameter importance but overlook parameter interference. The
central handshake panel represents the conceptual
unification introduced by DRIFT-MEDIAN.

## M Case Study

**Initial Meme (Semantic Fidelity: 3; Clarity: 2;**
**Engagement: 1)**: As shown in Figure6a, the
meme uses the*Batman slapping Robin*template to
humorously highlight the contrast between current
and earlier research approaches. The central idea is
that modern research employs more advanced techniques (e.g., Volume Sparse Deep Belief Networks
optimized with Neural Architecture Search), while
earlier methods (such as LSTMs and traditional
DBNs) are portrayed as outdated or complacent.

---

However, since the meme template depicts physical aggression, it can be interpreted as offensive
toward prior work. Moreover, the meme oversimplifies the nuances of the challenges faced by earlier methods and does not fully convey the specific
improvements in operational efficiency and adaptability introduced by the newer approach.

**Meme at 3rd Iteration (Semantic Fidelity: 4;**
**Clarity: 3; Engagement: 4)**: As illustrated in
Figure6b, the refined meme effectively captures
the paper’s main contribution—introducing the VS-
DBN architecture—and clearly communicates its
improvements in both accuracy and efficiency. It
also resolves the issue of offensiveness by adopting
a more suitable meme template that conveys the
intended message respectfully.

We use Volume Sparse Deep Relief Networks optimized with Meral Architecture Search to boost accuracy to 98.42%

But we're using LSTM and DBNs! They're good enough, right?

Current Research

Previous Research

Research Evolution

New Contributions

Volume-Sparse Deep Relief Network (V5-DRN) with Neural Architecture Search achieving 98.42% accuracy!

Prior Work

Standard DRNs and LSTMs struggling with overfitting and low accuracy...15

(a) Initial meme using an of-(b) Refined meme using a nonfensive template. offensive template.

Figure 6: Comparison between the initial and refined
memes: (a) the original meme that conveys the research
contrast but uses an offensive template; (b) the improved
meme that preserves the message while maintaining
respectfulness.

## M.1 Single-Agent CoT Prompt

We use the following rubric-guided prompt for the
single-agent baseline.

## Single-Agent CoT Meme Generator (Sys- tem Prompt)

You are an expert scientific communicator
and creative meme writer. Your goal is to
generate a scientific meme that is faithful to
the paper, easy to interpret, and engaging.
**Input.**You are given: (i) a paper abstract,
(ii) the full paper text when available, (iii) a
list of candidate meme templates, and (iv)
an evaluation rubric.
**Evaluation Criteria.**

• **Scientific Fidelity:** The meme must
accurately reflect the paper’s contribu-

tion without introducing unsupported
claims.

• **Clarity & Interpretability (FRI):**
The meme should be easy to understand for a broad research audience.

• **Engagement Potential:** The meme
should be memorable, humorous, and
well-matched to the selected template.

## Instructions.

1. Read the abstract and full paper (if
available). Identify: (a) the main limitation(s) of prior work, and (b) the core
contribution of the current paper.

2. Write a contrastive summary (at most
100 words) describing: (a) what prior
work did, (b) what this paper does differently, and (c) why the difference
matters.

3. Select the meme template that best captures this contrast.

4. Generate a meme caption for the selected template that is: (a) scientifically faithful, (b) clear and interpretable, and (c) engaging.

5. When there is a trade-off, prioritize:
Scientific Fidelity *>* Clarity & Interpretability*>*Engagement Potential.

## Output Format.

•**Prior-work limitations:**
<text>

•**Core contribution:**
<text>

• **Contrastive summary (***≤***100**
**words):**
<text>

•**Selected meme template:**
<template name>

•**Template selection rationale:**
<text>

•**Meme caption:**
<text>

---

**Constraints.**Do not introduce claims not
supported by the paper. Do not use humor
that distorts the scientific meaning. If only
the abstract is available, rely solely on the
abstract.

**M.2 Computational Cost Analysis**

| Method | API Calls | Total Tokens | Time (s) |
| --- | --- | --- | --- |
| Self-Reflect | 3 | 2,314 | 5.49 |
| Self-Refine | 9 | 10,873 | 24.31 |
| ChatEval (N=2) | 12 | 15,642 | 32.78 |
| SciMemeX | 14 | 19,284 | 54.12 |
| MAD (N=2) | 16 | 21,357 | 60.44 |

Table 6: Computational cost comparison per sample.
We report the average number of API calls, total tokens
(prompt + completion), and wall-clock time (seconds)
across 50 randomly selected samples.

Following reviewer suggestions, we analyze the
computational cost of all methods. We randomly
sample 50 papers from the evaluation set and measure the average cost per sample. For each method,
we log: (i) the number of API calls, (ii) the total number of tokens consumed (including both
prompt and completion tokens), and (iii) the wallclock execution time in seconds.

Table6reports the averaged results. As expected,
multi-agent and iterative methods incur higher computational cost than simpler baselines. SciMemeX
requires more API calls and tokens than singleagent approaches such as Self-Reflect, but remains
comparable to or more efficient than other multiagent frameworks such as MAD.

We note that the increased computational cost
of SciMemeX stems from its structured decomposition and iterative refinement process, which
contribute to improved performance in both automatic and human evaluations. This highlights
a trade-off between computational efficiency and
generation quality. Importantly, despite the higher
cost, SciMemeX achieves consistent gains across
heterogeneous backbone models and human evaluation, suggesting that the additional computation is
justified by improved output quality.