---
title: "Speculative Ensemble: Fast Large Language Model Ensemble via Speculation"
collection: publications
category: conferences
permalink: /publication/2025-02-01-speculative-ensemble
excerpt: 'Speculative Ensemble is a novel framework that accelerates the ensemble of any number of LLMs without sacrificing performance. It could reach 1.11x-2.23x over standard ensemble techniques on two-model or three-model pairs.'
date: 2025-02-01
venue: 'Arixv'
paperurl: 'https://arxiv.org/abs/2502.01662v1'
citation: 'Fu J, Jiang Y, Chen J, et al. Speculative Ensemble: Fast Large Language Model Ensemble via Speculation[J]. arXiv preprint arXiv:2502.01662, 2025.'
---


# 1. Introduction: The Need for Efficient Ensemble Inference  
While ensemble methods combining multiple Large Language Models (LLMs) enhance prediction robustness and accuracy, their computational demands create significant deployment challenges. Traditional approaches requiring \\(O(nT)\\) operations for \\(n\\) models generating \\(T\\) tokens prove impractical for real-time applications due to linear scaling of latency with model count.

We present **Speculative Ensemble (SE)**, a novel framework that synergizes speculative decoding with ensemble theory to achieve **1.11×–2.23×** speedups over conventional ensembles while preserving output quality. Our key innovation lies in reimagining the proposal-verification paradigm to enable parallelized token generation while maintaining statistical equivalence to standard ensemble distributions.

# 2. Background: From Speculative Decoding to Ensemble Acceleration 
<div class="figure" id="fig1"> 
<img src="http://kamichanw.github.io/speculative-ensemble/fig1.png" alt="Paper Figure 1"  />
<p>Comparison of (a) vanilla ensemble, (b) speculative decoding, and (c) speculative ensemble. In (b) and (c), each discrete blue block represents a probability calculated by one forward pass of \(\mathcal{M}_q\), while the continuous green block indicates the joint distribution requires only one forward pass of \(\mathcal{M}_p\).</p>
</div>

## 2.1 Limitations of Vanilla Ensembles  
Standard ensemble methods compute token distributions through weighted combinations:  
\\[
r_i(x) = \sum_{k=1}^n \lambda_k p_i^{(k)}(x)
\\]  
This sequential process (<a href="#fig1">Figure 1(a)</a>) necessitates n×T model invocations, creating fundamental latency barriers.

## 2.2 Speculative Decoding Revisited 
Speculative decoding (SD) accelerates autoregressive generation by leveraging two models:  
1. **Proposal Model** (\\( \mathcal{M}_q \\)): A lightweight model rapidly generates candidate tokens.  
2. **Target Model** (\\( \mathcal{M}_p \\)): A larger model verifies proposals in parallel.  

For a proposal sequence \\( x_{i+1}, \ldots, x_{i+\gamma} \sim \prod_{j=1}^\gamma q_{i+j}(x) \\), the target model computes verification distributions \\( p_{i+j}(x) \\). Tokens are accepted if:  
\\[
u_j \leq \min\left(1, \frac{p_{i+j}(x)}{q_{i+j}(x)}\right), \quad u_j \sim \mathcal{U}(0,1)
\\]  
Rejected tokens trigger resampling from \\( \text{norm}(\max(0, p_{i+j} - q_{i+j})) \\). This process is depicted in <a href="#fig1">Figure 1(b)</a>.

# 3. Speculative Ensemble 
## 3.1 A Two-fold Innovation
Speculative ensemble (SE) is based on the following two insights:
### Ensemble Distribution as Verification Criterion  
Speculative decoding (SD) allows not only sampling from the target model's distribution, but also sampling from the ensemble distribution of the proposal model and target model. Adopted this,
SE redefines the verification step of SD to sample from the **ensemble distribution** \\( r_i(x) \\) rather than a single target model, as illustrated in <a href="#fig1">Figure 1(c)</a>. For a two-model ensemble:  
\\[
r_i(x) = \lambda q_i(x) + (1-\lambda) p_i(x)
\\]  
The acceptance criterion becomes:  
\\[
u_j \leq \min\left(1, \frac{r_{i+j}(x)}{q_{i+j}(x)}\right)
\\]  

### Alternating Proposal Framework 
<div class="figure" id="fig2"> 
<img src="http://kamichanw.github.io/speculative-ensemble/fig2.png" alt="Paper Figure 2"  />
<p>The sketch of Alternate Proposal Framework. A continuous colored block indicates a single model invocation, with the bonus token highlighted in a red rounded box. Beginning from Step 2, \(\mathcal{M}_q\) and \(\mathcal{M}_p\) are invoked alternately. Each invocation involves both the verification of the current token and the generation of a bonus token. For clarity, we assume that the proposal length for each model is 1 and that all proposed tokens are accepted.</p>
</div> 

In standard SD, the proposer and verifier are fixed, with one model consistently serving as the proposer and the other as the verifier, which fails to fully leverage the bonus token. In SD, when all tokens from the proposal model are accepted by the target model, the target model will naturally generate an additional token, referred to as the **bonus token**. However, since the bonus token is drawn from the target model's distribution rather than the ensemble distribution, it cannot be directly appended to the ensemble output. Therefore, we propose an alternative approach: treating the bonus token as a proposal from the target model, which is then verified by the proposer model. This insight leads to the development of a more efficient framework, the **Alternate Proposal Framework**, illustrated in <a href="#fig2">Figure 2</a>:  
   1. \\( \mathcal{M}_q \\) proposes \\( \gamma_q \\) tokens verified by \\( \mathcal{M}_p \\).  
   2. If all tokens are accepted, \\( \mathcal{M}_p \\) generates a bonus token treated as \\( \mathcal{M}_p \\)’s proposal.  
   3. \\( \mathcal{M}_q \\) verifies this bonus token, creating a feedback loop.  

## 3.2 Generalization to \\( n \\)-Model Ensembles  
<div class="figure" id="fig3"> 
<img src="http://kamichanw.github.io/speculative-ensemble/fig3.png" alt="Paper Figure 3"  />
<p>The sketch of SE in three-model ensemble scenario. 
    The colored boxes represent the stored probability distributions, while the grey boxes represent the discarded ones. Each invocation involves scoring the current proposal tokens and generating a bonus token. For clarity, we assume that the proposal length for each model is 1 and that all proposed tokens are accepted.</p>
</div> 

In this subsection, we extend SE to the \\(n\\)-model ensemble scenario. The core principles remain similar to the two-model case, with acceleration driven by two key factors. First, each model can <span title="We use the term scoring rather than verification because, unlike in the two-model case, scoring does not immediately trigger verification; instead, verification occurs only after all models have scored a token."><i><b>score</b></i></span> the proposals of other models in parallel, where scoring refers to computing the probability distribution of a proposal from other models. Second, during scoring, a model can naturally generate a bonus token, which further improves efficiency. 

As shown in <a href="#fig3">Figure 3</a>, the process begins in step 1 with the default proposal model, \\(\mathcal{M}_1\\), generating a proposal token \\(x_1\\). In step 2, \\(\mathcal{M}_2\\) scores \\(x_1\\) while simultaneously generating a bonus token \\(x_2\\). Similarly, in step 3, \\(\mathcal{M}_3\\) scores both \\(x_1\\) and \\(x_2\\) in parallel and produces another bonus token, \\(x_3\\). At this point, \\(x_1\\) has been scored by both \\(\mathcal{M}_2\\) and \\(\mathcal{M}_3\\), enabling the computation of its ensemble distribution \\(r_1(x)\\) for verification. The associated distributions \\(p_1^{(1)}(x)\\), \\(p_2^{(1)}(x)\\), \\(p_3^{(1)}(x)\\) are no longer needed and are discarded.  

If \\(x_1\\) is accepted, \\(\mathcal{M}_1\\) computes \\(p_2^{(1)}(x)\\), \\(p_3^{(1)}(x)\\), \\(p_4^{(1)}(x)\\) in parallel as shown in step 5, allowing verification of \\(x_2\\). Otherwise, if \\(x_1\\) is rejected, all stored distributions are cleared, and \\(\mathcal{M}_1\\) generates a new proposal, similar to step 1. 

Experiments show 1.27×–1.85× speedups for 3-model ensembles on code generation tasks.  

# 4. Experimental Results  
## 4.1 Experimental Configuration  
We test SE across multiple tasks including code generation, mathematical reasoning, multi-task understanding, and text summarization on HumanEval, GSM8K, MMLU and CNNDM, respectively. 

Two ensemble functions were tested:  
1. Weighted Ensemble (WE):  
   \\[
   r(x) = \lambda q(x) + (1-\lambda)p(x)
   \\]  
   - Two-model: \\(\lambda = 0.5\\), \\(T = 1\\)  
   - Three-model: Equal weights (\\(1/3\\))  

2. Contrastive Decoding (CD):  
   \\[
   r(x) = \text{Softmax}(l_p - \mu l_q)
   \\]  
   - \\(\mu = 0.1\\), \\(T \in \\{0, 1\\}\\)  

Among two ensemble functions, three methods are compared: (1) the
standard ensemble (**WE**, **CD**); (2) an accelerated version
with speculative decoding (**SD**), using the smallest model as
the proposal and the ensemble model as the target (**WE-SD**,
**CD-SD**); and (3) Speculative Ensemble (**WE-SE**, **CD-SE**).

We experiment on different types of LLMs, shown as below.
<table class="model-config">
  <thead>
    <tr>
      <th>Ensemble Type</th>
      <th>Model Pair</th>
      <th>Proposer (\(\mathcal{M}_q\))</th>
      <th>Verifier (\(\mathcal{M}_p\))</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">WE</td>
      <td>Llama-Vicuna</td>
      <td>Llama-2-7B</td>
      <td>Vicuna-7B-V1.5</td>
    </tr>
    <tr>
      <td>Qwen-3b</td>
      <td>Qwen2.5-3B-Instruct</td>
      <td>Qwen2.5-Coder-3B-Instruct</td>
    </tr>
    <tr>
      <td>Qwen-1.5b</td>
      <td>Qwen2.5-1.5B-Instruct</td>
      <td>Qwen2.5-Coder-1.5B-Instruct</td>
    </tr>
    <tr>
      <td rowspan="3">CD</td>
      <td>Llama-3</td>
      <td>Llama-3.2-1B</td>
      <td>Llama-3.1-8B-Instruct</td>
    </tr>
    <tr>
      <td>Llama-2</td>
      <td>Llama-68M</td>
      <td>Llama-2-7B</td>
    </tr>
    <tr>
      <td>OPT</td>
      <td>OPT-125M</td>
      <td>OPT-13B</td>
    </tr>
  </tbody>
</table>


## 4.2 Main Results  
**Weighted Ensemble (WE) Performance**  
<table class="results">
  <thead>
    <tr>
      <th rowspan="2">Model Configuration</th>
      <th rowspan="2">Method</th>
      <th colspan="4">Speedup Factor</th>
    </tr>
    <tr>
      <th>HumanEval</th>
      <th>GSM8K</th>
      <th>MMLU</th>
      <th>CNNDM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">Llama-Vicuna</td>
      <td>WE</td>
      <td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td>
    </tr>
    <tr>
      <td>WE-SD</td>
      <td>1.27×</td><td>1.21×</td><td>1.19×</td><td>1.15×</td>
    </tr>
    <tr class="highlight">
      <td>WE-SE</td>
      <td><strong>1.58×</strong></td><td><strong>1.52×</strong></td><td><strong>1.41×</strong></td><td><strong>1.46×</strong></td>
    </tr>
    <tr>
      <td rowspan="3">Qwen-3b</td>
      <td>WE</td>
      <td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td>
    </tr>
    <tr>
      <td>WE-SD</td>
      <td>1.13×</td><td>1.06×</td><td>1.09×</td><td>1.08×</td>
    </tr>
    <tr class="highlight">
      <td>WE-SE</td>
      <td><strong>1.62×</strong></td><td><strong>1.52×</strong></td><td><strong>1.42×</strong></td><td><strong>1.38×</strong></td>
    </tr>
  </tbody>
</table>

**Contrastive Decoding (CD) Performance**  
<table class="results">
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Temp. (\(T\))</th>
      <th rowspan="2">Method</th>
      <th colspan="4">Speedup Factor</th>
    </tr>
    <tr>
      <th>HumanEval</th>
      <th>GSM8K</th>
      <th>MMLU</th>
      <th>CNNDM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="6">Llama-3</td>
      <td rowspan="3">0</td>
      <td>CD</td>
      <td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td>
    </tr>
    <tr>
      <td>CD-SD</td>
      <td>2.04×</td><td>1.81×</td><td>1.52×</td><td>1.58×</td>
    </tr>
    <tr class="highlight">
      <td>CD-SE</td>
      <td><strong>2.23×</strong></td><td><strong>2.00×</strong></td><td><strong>1.77×</strong></td><td><strong>1.61×</strong></td>
    </tr>
    <tr>
      <td rowspan="3">1</td>
      <td>CD</td>
      <td>1.00×</td><td>1.00×</td><td>1.00×</td><td>1.00×</td>
    </tr>
    <tr>
      <td>CD-SD</td>
      <td>1.55×</td><td>1.21×</td><td>1.20×</td><td>1.07×</td>
    </tr>
    <tr class="highlight">
      <td>CD-SE</td>
      <td><strong>1.65×</strong></td><td><strong>1.44×</strong></td><td><strong>1.31×</strong></td><td><strong>1.18×</strong></td>
    </tr>
  </tbody>
</table>

<style>
.model-config, .results {
  width: 100%;
  border-collapse: collapse;
  margin: 20px 0;
}
.model-config th, .model-config td, .results th, .results td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}
.results th {
  background: #f8f9fa;
}
.highlight {
  background-color: #e3f2fd;
}
.figure {
  text-align: center;
  margin: 20px 0;
}
.figure img {
  max-width: 80%;
  border: 
}
.figure p {
  text-align: left;
  font-size: small;
}
</style>
  
