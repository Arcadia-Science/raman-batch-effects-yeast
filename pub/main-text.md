# Leave-one-batch-out cross-validation reveals strong batch effects in Raman spectroscopy of yeast cultures

*We acquired spontaneous Raman spectra of several strains and species of yeast. We found that our data contains a mix of biological signals and experimental batch effects, and suggest that proper analysis of Raman spectra should include cross-validation on experimental replicates.*

## Purpose

At Arcadia, we are interested in using spontaneous Raman spectroscopy for biological phenotyping. Previous efforts have reported that Raman spectroscopy can be used to distinguish between different strains [ref](https://www.nature.com/articles/s41467-019-12898-9), cell types [ref](https://www.nature.com/articles/s41587-023-02082-2), and physiological states [ref](https://elifesciences.org/reviewed-preprints/101485). In an effort to obtain analogous results relevant to our own work, we acquired spontaneous spectra of two small collections of yeast strains, then trained standard ML models to predict both strain and species identity from the Raman spectra alone. We found that, in this dataset, experimental batch effects dominated strain-level biological signals but not species-level signals.

This work is primarily for researchers using Raman spectroscopy to detect meaningful differences between biological variables (e.g., species, strain, growth condition, cell type, cell state, etc.). We're sharing these results to call attention to the existence of experimental batch effects in spontaneous Raman spectra of biological samples, and of the consequent importance of rigorous experimental design and cross-validated analysis when using this technique.

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

The raw Raman spectra were processed using a standard pipeline based on the [`ramanspy`](https://github.com/barahona-research-group/RamanSPy) Python package. All data processing code is available in the GitHub repository accompanying this publication.

1. **Cosmic ray removal.** We applied the Whitaker-Hayes despiking algorithm with default parameters to remove cosmic ray artifacts.

2. **Background subtraction.** For each experimental replicate (plate), we acquired a dark spectrum from an empty region of the stainless steel plate. We computed a "consensus" background spectrum by averaging the dark spectra from all three plates, then subtracted this same consensus background spectrum from *all* sample spectra in the dataset. The use of a single background spectrum (rather than one per plate) is necessary to avoid injecting additional batch effects into the spectra from small per-plate differences in the background spectra.

3. **Smoothing and baseline correction.** After background subtraction, we applied a Savitzky-Golay filter to smooth the spectra, then subtracted the autofluorescence background using the ModPoly algorithm with polynomial order 5.

4. **Cropping and normalization.** After baseline correction, we cropped the spectra to the "fingerprint region" (300-1800 cm⁻¹) and normalized the cropped spectra using area-under-the-curve (AUC) normalization.

5. **Quality control to remove dim spectra.** The absolute, unnormalized intensity of some spectra was too low to yield useful Raman peaks. We identified and removed these "dim" spectra by applying an empirically determined threshold of 1000 intensity units to the mean intensity of the background-subtracted but otherwise unprocessed spectra. 

6. **Quality control to remove outlier spectra.** Within each well of each plate, we identified outlier spectra by computing the standard deviation of each spectrum's deviation from the group median spectrum, then used elbow detection on the sorted distances to identify outliers. This quality control removed a small fraction of spectra that likely resulted from incomplete desiccation or imaging artifacts.

7. **Batch correction.** To correct for plate-level batch effects, we applied a linear mixed model (LMM) independently to each wavenumber, treating plate identity as a random effect. We then crudely corrected for batch-specific scale factors by normalizing the residuals by their within-batch standard deviations. This approach assumes that all batches are sufficiently large and that the random effect (plate identity) is not confounded with the biological variable (strain identity) of interest.

### Analysis strategy

We took an ML-centric approach, treating each processed spectrum as a feature vector and the collection of all spectra as a feature matrix in which rows correspond to samples and columns to wavenumber. We trained random forest classifiers to predict strain identity and evaluated performance using two different cross-validation strategies:

1. **Standard k-fold cross-validation** in which all spectra are randomly partitioned into training and test sets. This is the standard approach to cross-validation, and is often used in the literature for datasets that are known or assumed to be homogeneous and well-balanced.
2. **Leave-one-plate-out cross-validation** in which spectra are partitioned into training and test sets according to the experimental replicate from which they came. In our case, we had three experimental replicates, so we partitioned all spectra from two plates into the training set, and all spectra from the held-out third plate into the test set.

We used the standard `scikit-learn` implementation of a random forest classifier, with 100 trees and class-weighting set to "balanced" to correct for class imbalances (which were generally minor, given our balanced experimental design). We used the default hyperparameters for all other settings.

## The results

After processing, we found that the spectra contained a few clear sharp peaks and many broad, diffuse features, and that the mean spectra for all strains and species looked similar (Figure 1). We therefore reasoned that carefully cross-validated modeling (rather than exploratory analysis) would be necessary to determine whether the spectra contained genuine biological signals.

### Standard cross-validation gives misleadingly good results

We first evaluated a strain classification task under standard k-fold cross-validation. The model performed very well (Figure 1). The confusion matrix showed a strong diagonal, indicating that the model correctly predicted strain identity across most samples.

[Figure 2: Confusion matrix from standard k-fold cross-validation showing strong diagonal]

### Leave-one-plate-out cross-validation reveals batch effects

We then evaluated the same task under leave-one-plate-out cross-validation. The results were significantly worse (Figure 2); the confusion matrix showed that only a few strains remained distinguishable. This result implies that plate-level batch effects dominate whatever strain-level signal exists in the spectra, and that the standard cross-validation approach was effectively overfitting to these batch-specific features. This is possible because each fold in the k-fold cross-validation procedure includes spectra from all three plates, so the model can "see" batch-specific features and use them to help predict strain identity.

[Figure 3: Confusion matrix from leave-one-plate-out cross-validation showing poor performance]

### Leave-one-strain-out cross-validation reveals a strong plate-level batch effect

We confirmed the existence of a plate-level batch effect by inverting the prediction and cross-validation dimensions: we trained a classifier to predict *plate identity* instead of strain identity, using leave-one-*strain*-out cross-validation. We found that the model could very reliably predict the plate from which each spectrum came. Since the plates correspond to end-to-end replicates of the same experimental protocol, there should be no "true" biological differences between the plates, implying the presence of strong plate-level batch effects that the model is able to exploit.

[Figure 4: Confusion matrix showing successful plate prediction with uncorreceted data and failed plate prediction after batch correction]

### Batch correction may sometimes help

We applied a linear mixed model to correct for plate-level effects on a per-wavenumber basis (TODO: add link to code). After correction, the "adversial" task to predict plate identity no longer worked (Figure XXX), confirming that the plate-level batch effect had been removed. However, strain-level classification was not improved (Figure 3B). This likely reflects some combination of 1) stochastic sample-level batch effects that are independent of experimental replicates and 2) genuinely subtle differences between the strains in our dataset that may not result in detectable Raman signatures.

[Figure 5: Confusion matrix showing strain prediction after batch correction]

### Species-level classification works well with or without batch correction

When we shifted from strain-level to species-level classification (*S. cerevisiae* vs. *S. pombe*), the model performance improved significantly (Figure 5). This was true with or without correcting for plate-level batch effects. This suggests that the spectra contain genuine species-level biological differences that are stronger than the experimental batch effects. Indeed, there was a hint that this was the case in our original strain-level confusion matrix (Figure 1); we can see that the misclassifications between strains were predominantly between strains of the same species.

[Figure 6: Mean spectra for *S. cerevisiae* and *S. pombe* with feature importance overlay showing wave numbers where species differ]

Finally, as an additional sanity check, we plotted the out-of-bag estimates of feature importance from the random forest classifier, and observed that the most important features aligned with wavenumbers at which the mean spectra visibly differed between species. This again suggests that the model is leveraging true differences between species that are stronger than the experimental batch effects.

## Conclusions

Raman spectroscopy is an extremely sensitive technique. It can detect real, relevant biological signals—-and sometimes very subtle ones—-but it also readily picks up irrelevant signals associated with experimental conditions. In practice, we have found that Raman spectra invariably contain a mixture of both relevant and irrelevant signals. Because experimental batch effects are a kind of structured noise, standard k-fold cross-validation can give misleadingly good results by allowing models to leverage batch-specific features. To distinguish authentic biological signals from batch effects, experiments must therefore include end-to-end replicates, and analysis must be cross-validated on experimentally meaningful batch dimensions like plate or replicate identifiers.

## Next steps

This work highlights the need for careful experimental design and validation strategies when using Raman spectroscopy for biological classification tasks. We recommend that experiments incorporate at least two, and ideally three, end-to-end replicates from the outset, and that analysis workflows include both leave-one-replicate-out cross-validation and "adversial" prediction tasks in which a model is trained to predict biologically meaningless variables like replicate or plate identifiers. Finally, we're also interested in exploring more sophisticated batch correction methods and understanding what types of biological differences tend to produce robust, generalizable Raman signatures versus those that are confounded by experimental variation.
