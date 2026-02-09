# Leave-one-batch-out cross-validation reveals strong batch effects in Raman spectroscopy of yeast cultures

*Standard k-fold cross-validation can give misleadingly good results when analyzing Raman spectra. We show that batch effects from experimental replicates can dominate biological signals, and proper validation requires cross-validating across experimentally meaningful dimensions like plate ID or replicate ID.*

## Purpose

At Arcadia, we are interested in using spontaneous Raman spectroscopy for biological phenotyping. Previous efforts [TODO: add citations] have reported that Raman spectroscopy can be used to distinguish between different strains of microorganisms, different cell types, and different . In an effort to reproduce these results, we acquired spontaneous spectra of two small collections of yeast strains (see table) that we were already working with for other purposes.

We found that Raman spectra contain a mix of experimentally structured noise (batch effects) and presumptively genuine biological signal.

We performed this analysis to understand why Raman-based classification models sometimes fail to generalize across experiments.

The key takeaway is that standard k-fold cross-validation can mask batch effects, leading to overconfident models that don't generalize.

This work is primarily for researchers using Raman spectroscopy to distinguish between biological samples or to predict sample identity. We're sharing it to call attention to the existence (and likely ubiquity) of experimental batch effects in spontaneous Raman spectra of biological samples and the consequent importance of rigorous experimental design and cross-validated analysis when using this technique.

All associated **data** and **code** is available in [this GitHub repository](TODO ADD LINK).

## Background and goals

Raman spectroscopy offers a promising approach for distinguishing between biological samples, as it is label-free, requires minimal sample preparation, and may detect subtle compositional differences that would otherwise require more expensive or laborious assays. A common analytical framework treats Raman spectra as feature vectors and uses machine learning models to predict sample identity. However, biological samples—-especially whole cells or organisms—-present challenges that may not be immediately apparent from standard validation metrics.

Our goal was to evaluate whether Raman spectra can be used to distinguish between different strains and species of yeast and to understand the extent to which batch effects from experimental variation might confound these predictions.

## The approach

### Experimental design

We collected Raman spectra from nine yeast strains, including wild-type and mutant strains from both *Saccharomyces cerevisiae* and *Schizosaccharomyces pombe*. We generated three "end-to-end" replicates by repeating the sample preparation and imaging protocols in triplicate, each with different cell cultures, physical plates, and imaging dates.

### Species and strains

TODO: add table of strains.

### Sample preparation

Strains were cultured in [TODO: add medium and conditions]. Saturated overnight cultures were spotted onto stainless steel plates and allowed to desiccate at room temperature and pressure.

### Raman spectroscopy

Spontaneous Raman spectra of the desiccated samples were acquired using our "InstantRaman" Raman spectrometer.

TODO: Add details about the Raman instrument and acquisition parameters.

### Data processing

The raw Raman spectra were processed using a standard pipeline based on the [`ramanspy`](TODO: add link) Python package.

TODO: explain processing.

### Analysis strategy

We took an ML-centric approach, treating each spectrum as a feature vector and the collection of spectra as a feature matrix. We trained random forest classifiers to predict strain identity and evaluated performance using two different cross-validation strategies:

1. **Standard k-fold cross-validation**: Randomly partitioning all spectra into training and test sets, regardless of which plate they came from
2. **Leave-one-plate-out cross-validation**: Training on spectra from two plates and testing on the third, then rotating through all three plates

We also performed an "adversarial" test by training models to predict plate identity—something we shouldn't be able to predict if there were no batch effects.

TODO: Add specific software versions and tools used for analysis.

## The results

### Standard cross-validation gives misleadingly good results

When we applied standard k-fold cross-validation to our strain classification task, the model performed very well. The confusion matrix showed a strong diagonal, indicating that the model correctly predicted strain identity across most samples.

[Figure 1: Confusion matrix from standard k-fold cross-validation showing strong diagonal]

### Leave-one-plate-out cross-validation reveals batch effects

When we switched to leave-one-plate-out cross-validation, the confusion matrix looked dramatically worse. The model trained on two plates largely failed to correctly classify samples from the held-out third plate, with only a few strains remaining distinguishable.

[Figure 2: Confusion matrix from leave-one-plate-out cross-validation showing poor performance]

This result implies that plate-level batch effects dominate the signal, and the standard cross-validation approach was effectively overfitting to these batch-specific features.

### Leave-one-strain-out cross-validation reveals a strong plate-level batch effect

We confirmed this interpretation by training a classifier to predict plate identity instead of strain identity. Despite there being no biological reason to distinguish plates, the model could very reliably predict which plate each spectrum came from. This adversarial test clearly demonstrates the presence of strong batch effects.

[Figure 3: Confusion matrix showing successful plate prediction]

### Batch correction methods help to some extent

We applied a linear mixed model to correct for plate-level effects. After correction, the model could no longer predict plate identity, confirming that the plate-level batch effect had been removed. However, this didn't substantially improve strain classification in this dataset.

The persistent difficulty with strain classification likely reflects some combination of: (1) sample-level batch effects not captured at the plate level, and (2) genuinely subtle differences between strains that may not produce strong Raman signatures.

### Species-level classification works well with or without batch correction

On a more optimistic note, when we shifted from strain-level to species-level classification (*S. cerevisiae* vs. *S. pombe*), the model performed very well with or without batch correction. This suggests that Raman spectroscopy can detect authentic biological differences when the signal is strong enough.

Notably, the misclassifications in the strain-level confusion matrix showed structure: mistakes were predominantly within species rather than between them, foreshadowing this result.

[Figure 4: Mean spectra for S. cerevisiae and S. pombe with feature importance overlay showing wave numbers where species differ]

The feature importance values from the random forest classifier aligned with wave numbers where the mean spectra visibly differed between species—a useful sanity check that the model is learning biologically meaningful features.

## Conclusions

Raman spectroscopy is extremely sensitive, which is both a blessing and a curse. It can detect real, relevant biological signals—sometimes very subtle ones—but it also readily picks up irrelevant signals associated with experimental design or protocol variation. In practice, it's almost always doing both. Standard k-fold cross-validation can give misleadingly good results by allowing the model to learn batch-specific features. To distinguish authentic biological signals from experimental artifacts, experiments must include end-to-end replicates, and analysis must be cross-validated on experimentally meaningful batch dimensions like plate or replicate ID.

## Next steps

This work highlights the need for careful experimental design and validation strategies when using Raman spectroscopy for biological classification tasks. Future experiments should incorporate end-to-end replicates from the outset and plan for leave-one-batch-out cross-validation as a standard practice. We're also interested in exploring additional batch correction methods and understanding what types of biological differences produce robust, generalizable Raman signatures versus those that remain confounded by experimental variation.
