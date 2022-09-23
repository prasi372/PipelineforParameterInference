# The effect of model levels and parameter inference settings towards negative feedback gene regulation
We present a pipeline for systematic comparison of model levels and parameter inference settings applied to negative feedback gene regulation. The article can be found on [bioRxiv][brx]. The data for the experiments is available [here][ArData].

## Summary

- Quantitative stochastic models of gene regulatory networks can be formulated at many different levels of fidelity. 
- It is challenging to determine what model fidelity to use in order to get accurate and representative results. 
- The model fidelity, the available data, and the numerical choices for inference and model selection - interplay in a complex manner. 
- We develop a computational pipeline designed to systematically evaluate inference accuracy for a wide range of true known parameters. We then use it to explore inference settings for negative feedback gene regulation.
- We compare a spatial stochastic model, a coarse-grained multiscale model, and a simple well-mixed model for several data-scenarios and for multiple numerical options for parameter inference. 
- This pipeline can potentially be used as a preliminary step to guide modelers prior to gathering experimental data. By training Gaussian processes to approximate the distance metric, we are able to significantly reduce the computational cost of running the pipeline.

## Code Organization
- The pipeline is presented in form of two notebooks present at the root of the repository.
- `scripts/experiments` consists of code used to generate date and conduct parameter inference experiments.
- `scripts/plotting` consists of code responsible to visualize results.
- `scripts/extract_model_metrics` consists of code responsible to compute various error/performance metrics.
- `scripts/Figs` consists of code used to produce figures in the article.

## License
MIT

## Contact
Marc Sturrock (marcsturrock@rcsi.ie)
Andreas Hellander (andreas.hellander@it.uu.se)

[//]: # 

   [brx]: <https://www.biorxiv.org/content/10.1101/2021.05.16.444348v2>
   [ArData]: <https://github.com/Aratz/MultiscaleCompartmentBasedModel>
