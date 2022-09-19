# Copyright 2022 The modeljoust Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Statistics for comparing accuracy of ML models.

The statistics implemented here are intended to compare the accuracy of multiple
ML models on the same set of examples. They test the null hypothesis that both
models attain equal accuracy on these examples against the alternative
hypothesis that one model is consistently better.

The `compare` function accepts per-example accuracies and performs either
approximate permutation tests or exact McNemar tests depending on whether
the per-example accuracies are real-valued or boolean. The permutation test
computes a p-value by comparing the observed mean accuracy difference to a
permutation distribution constructed by randomly exchanging the accuracy values
between the two models a specified number of times. For boolean per-example
accuracies (i.e., if the model is either correct or incorrect on each example),
the sign test corresponds to an exact permutation test that enumerates all
possible permutations. When used in this way, the sign test is equivalent to the
exact McNemar test.

The `get_per_example_accuracies` function provides a convenient way to
construct the per-example accuracies to be passed to the `compare` function,
given model predictions and true labels.
"""

import re
from typing import Any, List, Mapping, Sequence, Union
import warnings

import numpy as np
import scipy.stats

DEFAULT_NUM_PERMS = 10_000
Predictions = Union[np.ndarray,
                    Mapping[Any, Union[float, np.ndarray, List[Any]]]]


def _validate_inputs(a_accuracy: np.ndarray, b_accuracy: np.ndarray):
  """Validate inputs for statistical tests."""
  a_accuracy = np.array(a_accuracy, copy=False)
  b_accuracy = np.array(b_accuracy, copy=False)
  if a_accuracy.ndim != 1 or b_accuracy.ndim != 1:
    raise ValueError('a_accuracy and b_accuracy must be vectors.')
  if a_accuracy.size != b_accuracy.size:
    raise ValueError('a_accuracy and b_accuracy must have the same length.')
  return a_accuracy, b_accuracy


def sign_test(a_accuracy: np.ndarray, b_accuracy: np.ndarray,
              two_sided: bool = True) -> float:
  """Perform a sign test.

  When a_accuracy and b_accuracy are boolean, the sign test is equivalent to an
  exact McNemar test.

  Args:
    a_accuracy: An `num_examples` vector of per-example accuracies for model A.
    b_accuracy: An `num_examples` vector of per-example accuracies for model B.
    two_sided: Whether to perform a two-sided test. If False, the alternative
      hypothesis is that model A achieves higher accuracy than model B. If True,
      the alternative hypothesis is that models A and B achieve different
      accuracies.

  Returns:
    p-value of the exact McNemar test.
  """
  a_accuracy, b_accuracy = _validate_inputs(a_accuracy, b_accuracy)
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    return scipy.stats.binom_test(
        np.sum(a_accuracy > b_accuracy), np.sum(b_accuracy != a_accuracy),
        alternative='two-sided' if two_sided else 'greater')


def permutation_test(a_accuracy: np.ndarray, b_accuracy: np.ndarray,
                     two_sided: bool = True,
                     num_perms: int = DEFAULT_NUM_PERMS,
                     num_perms_per_chunk: int = 100,
                     seed: int = 1337) -> float:
  """Perform an exact McNemar test.

  Args:
    a_accuracy: An `num_examples` vector of per-example accuracies for model A.
    b_accuracy: An `num_examples` vector of per-example accuracies for model B.
    two_sided: Whether to perform a two-sided test. If False, the alternative
      hypothesis is that model A achieves higher accuracy than model B. If True,
      the alternative hypothesis is that models A and B achieve different
      accuracies.
    num_perms: Number of permutations to use, if a permutation test is used.
    num_perms_per_chunk: Number of permutations to compute at a time.
    seed: Seed to use for random number generator.

  Returns:
    p-value of the exact McNemar test.
  """
  a_accuracy, b_accuracy = _validate_inputs(a_accuracy, b_accuracy)

  # We can't subtract booleans, so make sure that a_accuracy is at least a
  # signed integer type.
  a_accuracy = a_accuracy.astype(np.promote_types(b_accuracy.dtype, 'i1'),
                                 copy=False)
  accuracy_diff = a_accuracy - b_accuracy
  # Can ignore examples where models are both equally accurate for efficiency.
  accuracy_diff = accuracy_diff[accuracy_diff != 0]
  true_sum_diffs = np.sum(accuracy_diff)
  if two_sided:
    true_sum_diffs = np.abs(true_sum_diffs)
  num_perms_computed = 0
  num_perms_geq = 0  # Number of permutations greater than the true difference.
  rng = np.random.default_rng(seed=seed)
  for i in range(0, num_perms, num_perms_per_chunk):
    n = min(num_perms_per_chunk, num_perms - i)
    num_perms_computed += n

    # Generate random signs for the difference.
    signs = 2 * rng.integers(low=0, high=2, size=(n, accuracy_diff.shape[0]),
                             dtype=np.int8) - 1
    # Create permutation values by randomly flipping signs, which is equivalent
    # to randomly swapping A and B.
    perm_values = np.sum(signs * accuracy_diff[None], -1)
    if two_sided:
      perm_values = np.abs(perm_values)
    num_perms_geq += np.sum(perm_values >= true_sum_diffs)

  # Ensure we compute the right number of permutations.
  assert num_perms_computed == num_perms
  return num_perms_geq / num_perms


def detectable_accuracy_delta(correct_agreement: float,
                              num_examples: int,
                              alpha: float = 0.05,
                              two_sided: bool = True) -> float:
  """Compute the minimum detectable accuracy delta.

  Given the proportion of examples that two models both get right/wrong,
  calculates the minimum accuracy delta that corresponds to a significant
  difference between them under a normal approximation.

  Args:
    correct_agreement: Proportion of examples that two models are both
      right or wrong (in [0, 1]).
    num_examples: Number of examples in the dataset.
    alpha: Desired significance threshold.
    two_sided: Whether to assume a two-sided test.

  Returns:
    The minimum accuracy delta that would result in a significant outcome.
  """
  n = (1 - correct_agreement) * num_examples
  # Obtained from https://en.wikipedia.org/wiki/Binomial_test#Large_samples
  # by setting p = 0.5 and solving for k. This gives the number of the
  # disagreements correctly classified by model A that would correspond to
  # a significant result. To convert to an accuracy delta, we then subtract
  # n/2 and multiply by 2.
  z = -scipy.stats.norm.ppf(alpha / (2 if two_sided else 1))
  example_delta = z * np.sqrt(n)
  return example_delta / num_examples


def compare(a_accuracy: np.ndarray, b_accuracy: np.ndarray,
            two_sided: bool = True,
            num_perms: int = DEFAULT_NUM_PERMS) -> float:
  """Perform statistical tests between models.

  This function compares the per-example accuracies of two models. If the
  arrays are boolean or 0/1, an exact McNemar test is used. Otherwise, an
  approximate permutation test, with `num_perms` perms, is used.

  Args:
    a_accuracy: An `num_examples` vector of per-example accuracies for model A.
    b_accuracy: An `num_examples` vector of per-example accuracies for model B.
    two_sided: Whether to perform a two-sided test. If False, the alternative
      hypothesis is that model A achieves higher accuracy than model B. If True,
      the alternative hypothesis is that models A and B achieve different
      accuracies.
    num_perms: Number of permutations to use for permutation tests (if
      permutation tests are used).

  Returns:
    A p-value for the null hypothesis that the models achieve the same accuracy,
    against the alternative hypothesis specified by the `two_sided` argument.
  """
  a_accuracy, b_accuracy = _validate_inputs(a_accuracy, b_accuracy)

  # If there are only two levels, then the permutation test is asymptotically
  # equivalent to a sign test. We expect the input will be 0/1 in this case.
  if np.all(((a_accuracy == 0) | (a_accuracy == 1)) &
            ((b_accuracy == 0) | (b_accuracy == 1))):
    return sign_test(a_accuracy.astype(bool, copy=False),
                     b_accuracy.astype(bool, copy=False),
                     two_sided=two_sided)
  else:
    return permutation_test(a_accuracy, b_accuracy, two_sided=two_sided,
                            num_perms=num_perms)


def get_per_example_accuracies(
    results: Sequence[Predictions], labels: Predictions,
    metric: str = 'top1') -> np.ndarray:
  """Compute metrics and perform statistical tests.

  Args:
    results: A sequence of per-example predictions. The predictions can be
      provided either as a mapping themselves, with example IDs as keys and
      lists of predictions as values, or as `num_examples` x `num_predictions`
      arrays. If the metric is "top1" or "mean_per_class", then predictions may
      be provided as scalars. Otherwise, predictions should be sorted in
      descending order of likelihood (i.e., top-1 first).
    labels: A vector, matrix, or mapping containing the correct labels,
      following the same format as `results`. If there are multiple labels for
      an example, all are treated as correct.
    metric: One of "topK" where K is an integer (e.g. top1 or top5) or
      "mean_per_class".

  Returns:
    An `num_examples x num_models` array of per-example accuracies. If `metric`
    is "topK", then accuracies are bools indicating whether each prediction is
    correct. If `metric` is "mean_per_class", then accuracies are zero for
    incorrectly predicted examples and non-zero for correctly predicted
    examples weighted so that the mean of the per-example accuracies gives the
    correct value of the per-class accuracy.
  """
  # Determine whether we've been given mappings or arrays.
  passed_mappings = isinstance(labels, Mapping)

  if metric == 'mean_per_class':
    topk_k = 1
  elif re.fullmatch('top[0-9]+', metric):
    topk_k = int(metric[3:])
  else:
    raise ValueError(
        f'Unrecognized metric "{metric}". Expected "topK", where K is an '
        'integer, or "mean_per_class".')

  per_example_accuracies = np.zeros((len(labels), len(results)), dtype=bool)

  if passed_mappings:
    key_set = set(labels.keys())
    key_list = list(labels.keys())
    labels_array = np.array(list(labels.values()))
    if labels_array.ndim == 1:
      labels_array = labels_array[:, None]
    elif labels_array.ndim > 2:
      raise ValueError(
          'Values of labels dict must be scalars, lists, or arrays.')

    for imodel, model_results in enumerate(results):
      if not isinstance(model_results, Mapping):
        raise ValueError(
            'If `labels` is provided as a mapping, values of `results` must '
            'also be mappings.')
      if set(model_results.keys()) != key_set:
        raise ValueError(f'Keys for model {imodel} do not match keys for '
                         'labels.')

      for iexample, example_key in enumerate(key_list):
        prediction = model_results[example_key]
        if topk_k == 1:
          if isinstance(prediction, Sequence):
            prediction = prediction[0]
          per_example_accuracies[iexample, imodel] = np.any(
              labels_array[iexample] == prediction)
        else:
          if not isinstance(prediction, Sequence) or len(prediction) < topk_k:
            raise ValueError('Predictions must be sequences containing at '
                             'least {topk_k} predictions.')
          per_example_accuracies[iexample, imodel] = np.any(
              labels_array[iexample][:, None] ==
              np.array(prediction[:topk_k])[None])
  else:
    labels_array = np.array(labels)
    if labels_array.ndim == 1:
      labels_array = labels_array[:, None]
    elif labels_array.ndim > 2:
      raise ValueError('Labels must be provided as a vector or matrix.')

    for imodel, model_results in enumerate(results):
      model_results_array = np.array(model_results)
      if model_results_array.ndim == 1:
        model_results_array = model_results_array[:, None]
      elif model_results_array.ndim != 2:
        raise ValueError(
            f'Predictions for model {imodel} are not provided as a '
            f'vector or matrix.')

      if model_results_array.shape[1] < topk_k:
        raise ValueError(
            'Asked to compute {metric} accuracy, but received only '
            f'{model_results_array.shape[1]} prediction(s) for model '
            f'{imodel}.')

      per_example_accuracies[:, imodel] = np.any(
          model_results_array[:, :topk_k, None] == labels_array[:, None, :],
          axis=(1, 2))

  if metric == 'mean_per_class':
    # Reweight accuracies so that their mean is the mean per class accuracy.
    _, label_index, label_count = np.unique(labels_array, return_inverse=True,
                                            return_counts=True)
    weights = len(labels_array) / (label_count[label_index] * len(label_count))
    per_example_accuracies = per_example_accuracies * weights[:, None]

  return per_example_accuracies
