# Leave-one-batch-out cross-validation reveals strong batch effects in Raman spectroscopy of yeast cultures

*We acquired spontaneous Raman spectra of several strains and species of yeast. We found that our data contains a mix of biological signals and experimental batch effects, and suggest that proper analysis of Raman spectra should include cross-validation on experimental replicates.*

## Purpose

At Arcadia, we are interested in using spontaneous Raman spectroscopy for biological phenotyping. Previous efforts [TODO: add citations] have reported that Raman spectroscopy can be used to distinguish between different strains [ref](https://www.nature.com/articles/s41467-019-12898-9), cell types [ref](https://www.nature.com/articles/s41587-023-02082-2), and physiological states [ref](https://elifesciences.org/reviewed-preprints/101485). In an effort to obtain analogous results relevant to our own work, we acquired spontaneous spectra of two small collections of yeast strains, then trained standard ML models to predict both strain and species identity from the Raman spectra alone. We found that experimental batch effects dominated strain-level, but not species-level, biological signals.

This work is primarily for researchers using Raman spectroscopy to distinguish between biological samples or states. We're sharing it to call attention to the existence of experimental batch effects in spontaneous Raman spectra of biological samples and of the consequent importance of rigorous experimental design and cross-validated analysis when using this technique.

All associated **data** and **code** is available in [this GitHub repository](TODO ADD LINK).

## Background and goals

Raman spectroscopy offers a promising approach for distinguishing between biological samples, as it is label-free, requires minimal sample preparation, and may detect subtle chemical-compositional differences that would otherwise require more expensive or laborious assays to measure. A common analytical framework treats Raman spectra as feature vectors and trains machine learning models to predict sample identity from held-out spectra. Our goal was to evaluate whether Raman spectra can be used to distinguish between different strains and species of yeast and to understand the extent to which batch effects from experimental variation confound these predictions.

## The approach

### Experimental design

We collected Raman spectra from nine yeast strains, including wild-type and mutant strains from both *Saccharomyces cerevisiae* and *Schizosaccharomyces pombe* (Table 1). We generated three "end-to-end" replicates by repeating the sample preparation and imaging protocols in triplicate, each with separate cell cultures, physical plates, and imaging dates.

### Species and strains

TODO: add table of strains.

### Sample preparation and Raman spectroscopy

TODO: add culture medium and culture conditions.

Saturated overnight cultures were spotted onto stainless steel plates and allowed to desiccate at room temperature and pressure. Spontaneous Raman spectra of the dessicated samples were acquired using our "InstantRaman" Raman spectrometer [TODO: add reference].

TODO: Add details about the Raman instrument and acquisition parameters.

### Data processing

The raw Raman spectra were processed using a standard pipeline based on the [`ramanspy`](TODO: add link) Python package.

TODO: explain processing steps.

### Analysis strategy

We took an ML-centric approach, treating each spectrum as a feature vector and the collection of all spectra as a feature matrix in which rows correspond to samples and columns to wavenumber. We trained random forest classifiers to predict strain identity and evaluated performance using two different cross-validation strategies:

1. **Standard k-fold cross-validation** in which we randomly partition all spectra into training and test sets.
2. **Leave-one-plate-out cross-validation** in which we partition spectra according to the experimental replicate from which they came.

TODO: Add specific software versions and tools used for analysis.

## The results

We found that our Raman spectra contain a mix of experimentally structured noise (that is, batch effects) and presumptively genuine biological signal.

### Standard cross-validation gives misleadingly good results

When we applied standard k-fold cross-validation to our strain classification task, the model performed very well. The confusion matrix showed a strong diagonal, indicating that the model correctly predicted strain identity across most samples.

[Figure 1: Confusion matrix from standard k-fold cross-validation showing strong diagonal]

### Leave-one-plate-out cross-validation reveals batch effects

When we switched to leave-one-plate-out cross-validation, the confusion matrix looked dramatically worse. The model trained on two plates largely failed to correctly classify samples from the held-out third plate, with only a few strains remaining distinguishable.

[Figure 2: Confusion matrix from leave-one-plate-out cross-validation showing poor performance]

This result implies that plate-level batch effects dominate the signal, and the standard cross-validation approach was effectively overfitting to these batch-specific features.

### Leave-one-strain-out cross-validation reveals a strong plate-level batch effect

We confirmed the existence of a plate-level batch effect by inverting the prediction and cross-validation dimensions: we trained a classifier to predict *plate identity* instead of strain identity, using leave-one-*strain*-out cross-validation. We found that the model could very reliably predict the plate from which each spectrum came. Since the plates correspond to end-to-end replicates of the same experimental protocol, there should be no biological differences between experiments. This adversarial test clearly demonstrates the presence of strong batch effects.

[Figure 3: Confusion matrix showing successful plate prediction with uncorreceted data and failed plate prediction after batch correction]

### Batch correction may sometimes help

We applied a linear mixed model to correct for plate-level effects on a per-wavenumber basis (TODO: add link to code). After correction, the "adversial" task to predict plate identity no longer worked (Figure XXX), confirming that the plate-level batch effect had been removed. However, strain-level classification was not improved (Figure 3B). This likely reflects some combination of 1) stochastic sample-level batch effects that are independent of experimental replicates and 2) genuinely subtle differences between the strains in our dataset that may not result in detectable Raman signatures.

[Figure 4: Confusion matrix showing strain prediction after batch correction]

### Species-level classification works well with or without batch correction

When we shifted from strain-level to species-level classification (*S. cerevisiae* vs. *S. pombe*), the model performed very well under leave-one-plate-out cross-validation, with or without batch correction. This suggests that Raman spectroscopy can detect authentic biological differences when the signal is stronger than the experimental batch effects. This result is, furthermore, consistent with our original strain-level model, as the misclassifications in the strain-level confusion matrix (Figure 1) were predominantly within species rather than between them.

[Figure 5: Mean spectra for S. cerevisiae and S. pombe with feature importance overlay showing wave numbers where species differ]

Finally, the out-of-bag estimates of feature importance from the random forest classifier aligned with wavenumbers at which the mean spectra visibly differed between species. This is an important sanity check that the model is learning biologically meaningful features.

## Conclusions

Raman spectroscopy is an extremely sensitive technique. It can detect real, relevant biological signals—-and sometimes very subtle ones—-but it also readily picks up irrelevant signals associated with experimental conditions. In practice, we have found that Raman spectra invariably contain a mixture of both. Because experimental batch effects are a kind of structured noise, standard k-fold cross-validation can give misleadingly good results by allowing models to leverage batch-specific features. To distinguish authentic biological signals from batch effects, experiments must therefore include end-to-end replicates, and analysis must be cross-validated on experimentally meaningful batch dimensions like plate or replicate identifiers.

## Next steps

This work highlights the need for careful experimental design and validation strategies when using Raman spectroscopy for biological classification tasks. Future experiments should incorporate at least two, and ideally three, end-to-end replicates from the outset, and the analysis workflow should include both leave-one-replicate-out cross-validation and "adversial" prediction tasks in which a model is trained to predict biologically meaningless variables like replicate or plate identifiers. We're also interested in exploring more sophisticated batch correction methods and understanding what types of biological differences produce robust, generalizable Raman signatures versus those that are confounded by experimental variation.
