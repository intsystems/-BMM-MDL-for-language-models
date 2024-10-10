# MDL for language model evaluation

[<img src="coverage-badge.svg">](https://github.com/intsystems/SoftwareTemplate-simplified/tree/master)
[<img src="https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white">](https://intsystems.github.io/SoftwareTemplate-simplified)

## Motivation
One of the method of language model analysis (and usage) is probing: we train a classifier for some specific layer of language model to extract some information about the analyzed words. For example, for PoS or syntactic properties of the words. The question we want to analyze is that which layer should we use, how should we choose this layer, and, generally, how much information about the downstream task can we capture from this layer/model. For this task multiple researchers proposed different theoretical frameworks. We propose to implement and compare different approaches used for this task.

## Key Works

The key works are:
1. T. Pimentel and R. Cotterell. [A Bayesian Framework for Information-Theoretic Probing](https://arxiv.org/abs/2109.03853.).
2. E. Voita and I. Titov. [Information-theoretic probing with minimum description length](https://arxiv.org/abs/2003.12298).
3. K. Stan ́czak, L.T. Hennigen, A. Williams, R. Cotterell, and I. Augenstein. [A latent-variable model for intrinsic probing](https://arxiv.org/abs/2201.08214).

## Members
1. [Anastasia Voznyuk](https://github.com/natriistorm)  (Project wrapping, Blog Post, Algorithm 1)
2. [Nikita Okhotnikov](https://github.com/Wayfarer123) (Library Wrapping, , Algorithm 2)
3. [Anna Grebennikova]() (Base code implementation, Demo completion, Algorithm 2)
4. [Yuri Sapronov](https://github.com/Sapr7) (Tests writing, Documentation Writing), Algorithm 3)

## Repository structure
```
problib/
├── __init__.py
├── utils.py
├── probing/
    ├── __init__.py
    ├── optimizer.py
    ├── mdl/
    ├── nn/
    ├── bayesian/
```

## Project structure


## Stack
**NLP Framework**: jiant, spaCy, Flair

**Basic code**: PyTorch

**Configs to interact with library**: YAML

**Bayesian instruments**: BayesPy

**Deploy**: HF Spaces, Gradio


## Master branch
By desing, master branch is protected from committing.  You should make pull requests to make changes into it.

## Documentation and test coverage
Documentation and test coverage badges can be updated automatically using [github actions](.github/workflows).

Initially both of these workflows are disabled (but can be run via "Actions" page).

To enable them automatically on push to master branch, change corresponding "yaml" files.
