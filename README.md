# modeljoust

modeljoust is a tool for computing whether one machine learning model performs
statistically significantly better than another.

## Usage

### Command-line

The included `compare_models.py` script reads text files containing labels and
model predictions and computes p-values for the comparison between each model
and the best. For example, to compare the accuracies of the ImageNet models for
which predictions are included, run:

```sh
python compare_models.py \
  --metric=top1 \
  --labels=predictions/imagenet/labels.txt \
  predictions/imagenet/models/*
```

This should produce the output:

```
| model                             |   top1 | p_value   |
|-----------------------------------|--------|-----------|
| model_soups_vit_g_best_holdout    |  90.72 | 0.0002    |
| model_soups_basic_best_holdout    |  90.83 | 0.002     |
| model_soups_vit_g_greedy_ensemble |  90.93 | 0.24      |
| model_soups_vit_g_greedy_soup     |  90.94 | 0.33      |
| model_soups_basic_greedy_soup     |  90.98 | 0.46      |
| model_soups_basic_greedy_ensemble |  91.02 | best      |
```

The provided p-values reflect statistical comparisons between each model and the
best. A low p-value indicates that the observed accuracy difference is unlikely
to arisen by chance. When reporting results in tables, we suggest choosing a
significance threshold (ɑ) of 0.05 and bolding accuracy numbers for all models
for which p-values are greater than this threshold, i.e., for which accuracies
are not statistically significantly different than the best.

Beyond computing p-values, `compare_models.py` also determines the number of
digits after the decimal point that are appropriate to report. To do so, it
computes minimum accuracy delta that would be statistically significant for a
model with the same correct/incorrect agreement as that observed between the
best model and the closest model to the best (in terms of correct/incorrect
agreement). It shows exactly enough digits that the accuracy number reported for
a model with this minimum delta from the best would be different from the
accuracy number reported for the best model. This means that the number of
digits reported can depend on the set of models being compared. When reporting
results in tables, we recommend using the same number of digits shown in the
table.

`compare_models.py` does not produce confidence intervals for the accuracy
because they are generally not useful for comparing different models on a fixed
test set. For models that make similar predictions, a standard binomial
confidence interval for the accuracy of one model will be much larger than the
minimum statistically detectable difference between the two models. In addition,
`compare_models.py` does not correct for multiple testing.

#### Model prediction file format

`compare_models.py` reads model predictions from CSV files. The first column
should be the example filename (or path, if the filename is not unique) and the
remaining columns are the top-k predictions in descending order.

Labels files are formatted similarly to model predictions, with example
filenames in the first column and correct labels in the remaining columns.

The included predictions files provide the top-5 predictions on
[ImageNet](https://www.image-net.org/),
[ImageNet-V2](https://github.com/modestyachts/ImageNetV2),
[ImageNet-R](https://github.com/hendrycks/imagenet-r),
[ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch),
[ObjectNet](https://objectnet.dev/), and
[ImageNet-A](https://github.com/hendrycks/natural-adv-examples),
for the models from
[Model Soups](https://arxiv.org/abs/2203.05482).

### API

#### `compare`

modeljoust's core function is
`compare(a_accuracy, b_accuracy, two_sided=True, num_perms=10_000)`. This
function accepts two vectors, `a_accuracy` and `b_accuracy`, containing the
per-example accuracies of two models on the same fixed test set. Often, the
vectors will contain bools indicating whether or not models classified each
example correctly, but `compare` also supports real-valued vectors. `compare`
performs a statistical test of the null hypothesis that the two models get the
same accuracy on the distribution from which the test set was drawn against the
alternative hypothesis that the models get different accuracy
(if `two_sided=True`) or model A gets higher accuracy than model B
(if `two_sided=False`).


#### `get_per_example_accuracies`

The function `get_per_example_accuracies(results, labels, metric='top1')`
provides a convenient way to construct the per-example accuracies to be passed
to the `compare` function, given model predictions and true labels. The results
and labels can be specified either as mappings from example IDs to lists of
predictions (sorted in descending order):

```python
modeljoust.get_per_example_accuracies(
  [
    {'example_1': [1, 2, 5],
     'example_2': [1, 5, 9]},
    {'example_1': [5, 3, 2],
     'example_2': [2, 1, 9]}
  ],
  {'example_1': 1, 'example_2': 2}
) # -> array([[ True, False],
  #           [False,  True]])
```

or as a `num_models x num_examples x num_predictions` of `results` and a vector
or `num_examples x num_correct_predictions` array of `labels`:

```python
modeljoust.get_per_example_accuracies(
  np.array([[[1, 2, 5],
             [1, 5, 9]],
            [[5, 3, 2],
             [2, 1, 9]]]),
  np.array([1, 2])
) # -> array([[ True, False],
  #           [False,  True]])
```

Valid metrics are "topK" for an integer K or "mean_per_class" for mean
per-class accuracy.

## Details

If per-example accuracies are boolean or 0/1, modeljoust performs a
[sign test](https://en.wikipedia.org/wiki/Sign_test), which is equivalent to an
[exact McNemar test](https://en.wikipedia.org/wiki/McNemar%27s_test). This test
rejects the null hypothesis if, on the examples for which exactly one of the two
models is correct, the accuracy of model A is significantly different from 50%.

If per-example accuracies are not boolean or 0/1, modeljoust performs a
permutation test to determine whether the difference in the mean accuracies
between the models is significantly different from the distribution given by
randomly exchanging the per-example accuracies of the two models. By default,
modeljoust uses 10,000 permutations. When per-example accuracies take on two
values, this permutation test is an approximation to the sign test that becomes
exact if all permutations are enumerated.

## Credits

The code in this repository was written following conversations with Jiahui Yu
and Ludwig Schmidt. The statistical approach matches
[Do Better ImageNet Models Transfer Better?](https://arxiv.org/abs/1805.08974)
and
[Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482).

If you find this code useful, please cite:

```
@misc{kornblith2022modeljoust,
  title = {modeljoust},
  author = {Kornblith, Simon},
  howpublished = "\url{https://github.com/google-research/modeljoust}",
  year = {2022}
}
```

## Contributing

We are happy to accept pull requests containing the predictions of new (or
existing) ImageNet models on the tasks for which we include predictions in the
`predictions/` subdirectory. We are also happy to accept pull requests
with predictions on other tasks where statistical comparison would be useful to
the ML community.

**This is not an official Google product.**