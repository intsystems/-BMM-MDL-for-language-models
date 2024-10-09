# MDL for language model evaluation

[<img src="coverage-badge.svg">](https://github.com/intsystems/SoftwareTemplate-simplified/tree/master)
[<img src="https://img.shields.io/badge/github%20pages-121013?style=for-the-badge&logo=github&logoColor=white">](https://intsystems.github.io/SoftwareTemplate-simplified)

## Motivation

## Key Works

The key works are:
1. T. Pimentel and R. Cotterell. [A Bayesian Framework for Information-Theoretic Probing](https://arxiv.org/abs/2109.03853.).
2. E. Voita and I. Titov. [Information-theoretic probing with minimum description length](https://arxiv.org/abs/2003.12298).
3. K. Stan ́czak, L.T. Hennigen, A. Williams, R. Cotterell, and I. Augenstein. [A latent-variable model for intrinsic probing](https://arxiv.org/abs/2201.08214).


## Repository creation
problib/
├── __init__.py
├── utils.py
├── probing/
    ├── __init__.py
    ├── optimizer.py
    ├── mdl/
    ├── nn/
    ├── bayesian/

## Master branch
By desing, master branch is protected from committing.  You should make pull requests to make changes into it.

## Documentation and test coverage
Documentation and test coverage badges can be updated automatically using [github actions](.github/workflows).

Initially both of these workflows are disabled (but can be run via "Actions" page).

To enable them automatically on push to master branch, change corresponding "yaml" files.
