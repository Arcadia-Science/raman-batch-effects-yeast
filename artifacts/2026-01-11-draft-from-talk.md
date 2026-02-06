# Leave-one-batch-out cross-validation reveals hidden batch effects in Raman spectroscopy of yeast

*Standard cross-validation of Raman spectra can give misleadingly good results. We show that batch effects from experimental replicates can dominate biological signals, and proper validation requires cross-validating on experimentally meaningful dimensions.*

## Purpose

We performed this analysis to understand why Raman-based classification models sometimes fail to generalize across experiments. The key takeaway is that standard k-fold cross-validation can mask batch effects, leading to overconfident models. This work is primarily for researchers using Raman spectroscopy (or other high-dimensional biological assays) for classification tasks. We're sharing it to help others avoid common pitfalls in model validation that we learned the hard way.

## Background and goals

Raman spectroscopy offers a promising approach for distinguishing between different biological samples, including microbial strains. A common analytical framework treats Raman spectra as feature vectors and uses machine learning classifiers to predict sample identity. However, biological samples—especially whole cells or organisms—present analytical challenges that may not be immediately apparent from standard validation metrics.

Our goal was to evaluate how well Raman spectra can distinguish between different yeast strains and to understand the extent to which batch effects from experimental variation might confound these classifications.

## The approach

### Experimental design

We collected Raman spectra from nine yeast strains, including wild-type and mutant strains from both *Saccharomyces cerevisiae* and *Schizosaccharomyces pombe*. The experiment was performed as three end-to-end replicates: different physical samples, different stainless steel plates, and imaging performed on different days.

### Sample preparation

Saturated overnight cultures were spotted onto stainless steel plates and allowed to desiccate. The dried cell samples were then analyzed using a Raman microscope.

**\[Author: please add details about the Raman instrument and acquisition parameters\]**

### Analysis strategy

We took an ML-centric approach, treating each spectrum as a feature vector and the collection of spectra as a feature matrix. We trained random forest classifiers to predict strain identity and evaluated performance using two different cross-validation strategies:

1. **Standard k-fold cross-validation**: Randomly partitioning all spectra into training and test sets, regardless of which plate they came from  
2. **Leave-one-plate-out cross-validation**: Training on spectra from two plates and testing on the third, then rotating through all three plates

We also performed an "adversarial" test by training models to predict plate identity—something we shouldn't be able to predict if there were no batch effects.

## Results

### Standard cross-validation gives misleadingly good results

When we applied standard k-fold cross-validation to our strain classification task, the model performed very well. The confusion matrix showed a strong diagonal, indicating that the model correctly predicted strain identity across most samples.

**\[Figure: Confusion matrix from standard k-fold cross-validation showing strong diagonal\]**

### Leave-one-plate-out cross-validation reveals batch effects

When we switched to leave-one-plate-out cross-validation, the confusion matrix looked dramatically worse. The model trained on two plates largely failed to correctly classify samples from the held-out third plate, with only a few strains remaining distinguishable.

**\[Figure: Confusion matrix from leave-one-plate-out cross-validation showing poor performance\]**

This result implies that plate-level batch effects dominate the signal, and the standard cross-validation approach was effectively overfitting to these batch-specific features.

### Confirming the batch effect

We confirmed this interpretation by training a classifier to predict plate identity instead of strain identity. Despite there being no biological reason to distinguish plates, the model could very reliably predict which plate each spectrum came from. This adversarial test clearly demonstrates the presence of strong batch effects.

**\[Figure: Confusion matrix showing successful plate prediction\]**

### Batch correction partially helps

We applied a linear mixed model to correct for plate-level effects. After correction, the model could no longer predict plate identity, confirming that the plate-level batch effect had been removed. However, this didn't substantially improve strain classification in this dataset.

The persistent difficulty with strain classification likely reflects some combination of: (1) sample-level batch effects not captured at the plate level, and (2) genuinely subtle differences between strains that may not produce strong Raman signatures.

### Species-level classification works well

On a more optimistic note, when we shifted from strain-level to species-level classification (*S. cerevisiae* vs. *S. pombe*), the model performed very well with or without batch correction. This suggests that Raman spectroscopy can detect authentic biological differences when the signal is strong enough.

Notably, the misclassifications in the strain-level confusion matrix showed structure: mistakes were predominantly within species rather than between them, foreshadowing this result.

**\[Figure: Mean spectra for S. cerevisiae and S. pombe with feature importance overlay showing wave numbers where species differ\]**

The feature importance values from the random forest classifier aligned with wave numbers where the mean spectra visibly differed between species—a useful sanity check that the model is learning biologically meaningful features.

## Key takeaways

Raman spectroscopy is extremely sensitive, which is both a blessing and a curse. It can detect real, relevant biological signals—sometimes very subtle ones—but it also readily picks up irrelevant signals associated with experimental design or protocol variation. In practice, it's almost always doing both. Standard k-fold cross-validation can give misleadingly good results by allowing the model to learn batch-specific features. To distinguish authentic biological signals from experimental artifacts, experiments must include end-to-end replicates, and analysis must be cross-validated on experimentally meaningful batch dimensions like plate or replicate ID.

## Next steps

This work highlights the need for careful experimental design and validation strategies when using Raman spectroscopy for biological classification tasks. Future experiments should incorporate end-to-end replicates from the outset and plan for leave-one-batch-out cross-validation as a standard practice. We're also interested in exploring additional batch correction methods and understanding what types of biological differences produce robust, generalizable Raman signatures versus those that remain confounded by experimental variation.

**\[Author: Are there specific batch correction methods or experimental designs you plan to test next? Consider adding details here.\]**  
