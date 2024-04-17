<p align="center">
    <img src="https://i.postimg.cc/vZPgPZZz/roc-curve-package.png" style="height: 450px;"/>
</p>

# Research Utility Package for ROC Curves and AUROC Analysis

<b>Fun Fact:</b> The term “Receiver Operating Characteristic” has its roots in <a href="https://www.statisticshowto.com/receiver-operating-characteristic-roc-curve/" target="_blank">World War II</a>. ROC curves were originally developed by the British as part of the <a href="https://en.wikipedia.org/wiki/Chain_Home" target="_blank">Chain Home</a> radar system. The analysis technique was used to differentiate between enemny aircraft and random noise.

## Table of Contents

#### [What are ROC Curves?](#what-are-roc-curves)

#### [What is AUROC analysis?](#what-is-auroc-analysis)

#### [Hanley and McNeil](#james-a-hanley-and-barbara-mcneil-methodology)

#### [Package Overview](#package-overview)

- [Getting Started](#getting-started)
- [Find Average Correlation between Two Models](#find-average-correlation-between-two-models)
- [Find Correlation Coefficient between Two Models](#find-correlation-coefficient-between-two-models)
- [Q1 and Q2 Calculations](#q1-and-q2-calculations)
- [Get T-Stat to Compare Two Models](#get-t-stat-to-compare-two-models)
- [Get Z-Score to Compare Two Models](#get-z-score-to-compare-two-models)
- [Get Non-Parametric AUROC Score](#get-non-parametric-auroc-score)
- [Get AUROC Bootstrapped P-Value to Compare Two Models](#get-auroc-bootstrapped-p-value-to-compare-two-models)
- [Create Stacked ROC Plot for Multiple Models](#create-stacked-roc-plot-for-multiple-models)
- [Get Optimal Classification Threshold from ROC Curve](#get-optimal-classification-threshold-from-roc-curve)
- [Get Optimal Classification Threshold from ROC Curve for Imbalanced Dataset](#get-optimal-classification-threshold-from-roc-curve-for-imbalanced-dataset)

## What are ROC Curves?

ROC, or Receiver Operating Characteristic Curve, is essentially a graph that shows how well a binary classification problem is performing. When observing the graph, there is a straight line cutting through the graph at a 45 degree angle. This line represents random guessing, i.e. the model is no better at classifying a class than a coin flip.

<p align="center">
    <a href="https://www.evidentlyai.com/classification-metrics/explain-roc-curve#how-to-get-the-roc-auc-score" target="_blank">
        <img src="https://assets-global.website-files.com/6266b596eef18c1931f938f9/647607123e84a06a426ce627_classification_metrics_014-min.png" style="height: 300px; margin: auto;"/>
    </a>
</p>
The lines can be interpreted as the trade-off between <a href="https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc" target="_blank">true positive rate (TPR) and the false positive rate (FPR)</a> . TPR is also often referred to as the <b>recall score</b>, especially in machine learning contexts. It is calculated by dividing the number of correctly identified true cases divided by the sum of the number of correctly identifies true cases and the false negatives.

$$
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
$$

On the x-axis is the FPR. It is the ratio of false positive predictions to the total number of actual negative instances.

$$
\text{FPR} = \frac{\text{False Positives}}{\text{False Positives} + \text{True Negatives}}
$$

The shows the tradeoff between these two metrics at different <b>thresholds</b> for classification.
The threshold simply meaning the cutoff point for a positive classification.

$$
f(X, \tau) = \begin{cases}
    1 & \text{if } X > \tau \\
    0 & \text{otherwise}
\end{cases}
$$

The way to interpret the graph visually is: the closer the graph's line is to the top left of the plot, the better the model is at making classifications. A perfectly classifying model would go straight from zero up to the top-left corner and then straight across the graph horizontally.

## What is AUROC analysis?

While visual inspection of the ROC plot is sufficient in certain cases, oftentimes it can be beneficial to dive deeper in the model's effectiveness. This is where AUROC, or <b>Area Under the Receiver Operating Characteristic</b>, analysis comes into play. It is exactly what it sounds like, in the sense that it is literally the area under the ROC curve. Conceptually, it is a single number which quantifies the overall ability of the model to classify beyond visual inspection alone. It is a score which ranges between 0 and 1, where 0.5 represents random guessing and 1 represents perfect performance. There are several different ways to calculate the AUROC score, but the most common method is to use the <a href="https://en.wikipedia.org/wiki/Trapezoidal_rule" target="_blank">trapezoidal rule</a>. This is not something anyone would ever calculate by hand, but the mechanics basically involve forming trapezoids progressively across the graph to estimate the area.

<p align="center">
    <a href="https://en.wikipedia.org/wiki/Trapezoidal_rule" target="_blank">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Trapezium2.gif/440px-Trapezium2.gif" style="height: 300px"/>
    </a>
</p>

## James A. Hanley and Barbara McNeil Methodology

Hanley, of <a href="https://www.mcgill.ca/epi-biostat-occh/james-hanley" target="_blank">McGill University</a>, and McNeil, of <a href="https://hcp.hms.harvard.edu/people/barbara-j-mcneil?page=18" target="_blank">Harvard University</a>, published a landmark paper titled: <a href="https://pubs.rsna.org/doi/epdf/10.1148/radiology.143.1.7063747" target="_blank">The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve</a> in 1982. The duo then followed this paper up in 1983 with another aptly titled landmark paper:
<a href="https://pubs.rsna.org/doi/epdf/10.1148/radiology.148.3.6878708" target="_blank">A Method of Comparing the Areas under Receiver Operating Characteristic Curves Derived from the Same Cases.</a> To summarize, they created a methodology for using the AUROC score to perform hypothesis tests in order to compare models in a statistically significant way using the Z-Score.

$$
\text{Z} = \frac{AUC_1 - AUC_2}{\sqrt{\frac{V_1 + V_2 - 2 \times \text{Cov}(AUC_1, AUC_2)}{2}}}
$$

Hanley and McNeil's methodology has since been adopted and modified by other researchers for applications outside of medicine and is now widely used for analyzing the performance of machine learning models.

# ROC Utils Package Methods

This section outlines the methods available with this package. The purpose of these functions is to compare two binary classification models. <span style="color: orange;">Whenever a method call for a y_pred parameter, this does <b>NOT</b> refer to the actual predictions of the model, but the probability of a positive classification.</span>

## Getting Started

Install and load this package like you would any other pip package.

```bash
pip install research-roc-utils
```

After installation, load in your python file.

```python
import research_roc_utils.roc_utils as ru
```

## Find Average Correlation between Two Models

This function takes y_true, y_pred_1, y_pred_2, and corr_method and returns the average correlation between the positive and negative classifications based on a passed correlation method. <b>IMPORTANT:</b> there is no default correlation method and you must pass in a correlation function. Unless you specifically need this value, it will be called internally by other methods so you will not need to call this method yourself.

```python
import research_roc_utils.roc_utils as ru
from scipy.stats import kendalltau

# avg_corr_fun(y_true, y_pred_1, y_pred_2, corr_method)

avg_corr = ru.avg_corr_fun(y_true, y_pred_1, y_pred_2, kendalltau)
```

## Find Correlation Coefficient between Two Models

This method find the correlation coefficient of two models based on the Hanley and McNeil table. This is commonly represented as <b>r</b> in their papers. The function takes two arguments: the average correlation between the two models for both the positive and negative classifications and the average AUROC between the models. For most methods, you don't need to implement this function yourself and it is called internally.

```python
import research_roc_utils.roc_utils as ru

r_value = ru.avg_corr_fun(avg_corr, avg_area)
```

## Q1 and Q2 Calculations

Unless you specifically need these values individually, you will not need to call this yourself. However, if you want to use these values independently, you can call this method.

```python
import research_roc_utils.roc_utils as ru

from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_true, y_pred)

q_1, q_2 = ru.q_calculations(roc_auc)
```

## Get T-Stat to Compare Two Models

This method for calculating the T-Stat is adopted from the paper <a href="https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr691.pdf" target="_blank">What Predicts U.S. Recessions? (2014)</a> by the researchers Prof. Weiling Liu and Prof. Dr. Emanuel Mönch of <a href="https://damore-mckim.northeastern.edu/people/weiling-liu/" target="_blank">Northeastern University</a> and the <a href="https://www.frankfurt-school.de/en/home/research/staff/Emanuel-Moench" target="_blank">Frankfurt School of Finance and Management</a> respectively. Their paper is very interesting and I highly recommend reading it if you are interested in learning more about the mechanics and potential application of the ROC T-test. Takes the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html" target="_blank">sklearn roc_auc_score</a> method for calculating AUROC by default. Returns a T-stat which you can use for hypothesis testing.

```python
import research_roc_utils.roc_utils as ru
from scipy.stats import kendalltau

# roc_t_stat(y_true, y_pred_1, y_pred_2, corr_method, roc_auc_fun=roc_auc_score)

t_stat = ru.roc_t_stat(y_true, model_1_y_pred, model_2_y_pred, kendalltau)
```

## Get Z-Score to Compare Two Models

This method is used to calculate the Z-Score based on the implementation of Hanley and McNeil outlined above. This score can be used for hypothesis testing and comparing two models. Also uses the
<a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html" target="_blank">sklearn roc_auc_score</a> as the default roc_auc_fun argument.

```python
import research_roc_utils.roc_utils as ru
from scipy.stats import pearsonr

# roc_z_score(y_true, y_pred_1, y_pred_2, corr_method, roc_auc_fun=roc_auc_score)

z_score = ru.roc_z_score(y_true, y_pred_1, y_pred_2, pearsonr)
```

## Get Non-Parametric AUROC Score

This method is useful in contexts such as econometric or financial analysis when it is not appropriate to assume a specific probability distribution of the data being analyzed. This methodology was adapted from the groundbreaking paper <a href="https://economics.ucr.edu/wp-content/uploads/2019/10/Jorda-paper-for-1-24-11-seminar.pdf" target="_blank">Performance Evaluation of Zero Net-Investment Strategies</a> by Òscar Jordà of the <a href="https://economics.ucdavis.edu/people/oscar-jorda" target="_blank">University of California at Davis</a> and Alan M. Taylor also of <a href="https://economics.ucdavis.edu/people/alan-taylor" target="_blank">University of California at Davis.</a> I highly recommend reading their widely cited paper for more information on the theory behind this method.

$$
\text{AUC} = \frac{1}{TN \times TP} \sum_{i=1}^{TN} \sum_{j=1}^{TP} \left( I(v_j < u_i) + \frac{1}{2} I(u_i = v_j) \right)
$$

```python
auc = ru.auroc_non_parametric(y_true, y_pred_prob)
```

## Get AUROC Bootstrapped P-Value to Compare Two Models

This method returns a p-value by comparing two models using bootstrapping. The method can be used to perform both one-sided and two-sided hypothesis tests. The structure for the code itself was inspired by another open source project <a href="https://github.com/mateuszbuda/ml-stat-util/tree/master" target="_blank">ml-stat-util</a> by <em><a href="https://github.com/mateuszbuda" target="_blank">@mateuszbuda</a></em> who is a machine learning researcher based out of Poland. This implementation uses a different methodology for calculating the p-value for two-sided tests, changes the bootstrapping methodology, uses differnt scoring functionality, and is optimized to work with AUROC scores specifically. Returns the p-value as well as a list, l, of the deltas between the two models' AUROC scores based on whatever compare_fun is passed in. Uses the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html" target="_blank">sklearn roc_auc_score</a> as the default score_fun argument.

```python
"""
p, l = boot_p_val(
    y_true,
    y_pred_1,
    y_pred_2,
    compare_fun=np.subtract,
    score_fun=roc_auc_score,
    sample_weight=None,
    n_resamples=5000,
    two_tailed=True,
    seed=None,
    reject_one_class_samples=True
    )
"""

import research_roc_utils.roc_utils as ru

# H0: There is no difference between Model 1 and Model 2
# H1: Model 2 is better than Model 1

p, l = ru.boot_p_val(y_true, model_1_y_pred, model_2_y_pred, n_resamples=10000, two_tailed=False)

if 1 - p < 0.05:
    print("Reject H0 with P-Val of: ", 1 - p)
else:
    print("Insufficient evidence to reject H0")

```

## Create Stacked ROC Plot for Multiple Models

This method uses the <a href="https://matplotlib.org/stable/index.html" target="_blank">matplotlib</a> library to easily create a stacked ROC plot of multiple models. Abstracts the functionality for creating the plot itself and returns a plot object with minimal styling which can then be further customized. Uses the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html" target="_blank">sklearn</a> auc method for calculating the AUROC for each model as the default auc_fun. The default roc_fun is from <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html" target="_blank">sklearn</a> for generating the TPR and FPR values.

- <b>model_preds:</b> array of prediction probabilities -> list of lists where each item is a model's y_pred values
- <b>model_names:</b> list of strings containing the names for each model in the same order as the model_preds variable
- <b>rand_guess_color:</b> color for the line splitting the plane of the graph representing a random guess

```python
"""
stacked_roc_plt(
        y_true,
        model_preds,
        model_names,
        roc_fun=roc_curve,
        auc_fun=auc,
        fig_size=(8,6),
        linewidth=2,
        linestyle='-',
        rand_guess_color='black'
        )
"""

import research_roc_utils.roc_utils as ru

plt_obj = ru.stacked_roc_plt(y_true, model_preds, model_names, rand_guess_color='darkred')
plt_obj.title("Stacked ROC Curve for 12 Month Lag")
plt_obj.legend(loc="lower right")
plt_obj.grid(True)
plt_obj.show()
```

## Output:

<p align="center">
    <img src="https://i.postimg.cc/8CWhR4hk/stacked-roc-curve-lag-12.png" style="height: 450px" target="_blank"/>
</p>

## Get Optimal Classification Threshold from ROC Curve

This method returns the optimal threshold based on the ROC curve based on the TPR and FPR. It works by finding the threshold where the delta between the FPR and TPR is maximized.

```python
import research_roc_utils.roc_utils as ru

threshold = ru.optimal_threshold(y_true, y_pred)
```

## Get Optimal Classification Threshold from ROC Curve for Imbalanced Dataset

This method returns the optimal threshold based on the ROC curve, but is optimized for imbalanced datasets. This method uses the geometric mean for finding the best threshold. Check out this link for more details on the implementation: <a href="https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/" target="_blank"></a>

```python
import research_roc_utils.roc_utils as ru

threshold = ru.optimal_threshold_imbalanced(y_true, y_pred)
```
