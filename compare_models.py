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

"""Tool for comparing models from the command line.
"""

import os

from absl import app
from absl import flags
import modeljoust
import numpy as np
from tabulate import tabulate


METRIC = flags.DEFINE_string(
    'metric', 'top1',
    'Metric to use. One of "topK" where K is an integer (e.g. top1 or top5) or '
    '"mean_per_class".')

LABELS = flags.DEFINE_string(
    'labels', None, 'Path to file containing the correct labels.')

TWO_SIDED = flags.DEFINE_bool(
    'two_sided', True, 'Whether to use a two-sided test.')


def main(argv):
  # Read labels.
  labels = modeljoust.read_predictions_file(LABELS.value)

  # Read model results.
  model_names = []
  model_names_set = set()
  results = []
  for model_file in argv[1:]:
    # Determine a unique name for this model.
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    if model_name in model_names_set:
      model_name = model_file

    model_names.append(model_name)
    model_names_set.add(model_name)
    results.append(modeljoust.read_predictions_file(model_file))

  per_example_accuracies = modeljoust.get_per_example_accuracies(
      results, labels=labels, metric=METRIC.value)
  mean_accuracies = np.mean(per_example_accuracies, 0)

  # Find models that are numerically the best. We need to account for some
  # floating point error in case the metric is mean_per_class.
  best_models = np.where(
      np.isclose(mean_accuracies, np.max(mean_accuracies),
                 0, 1e-3 / per_example_accuracies.shape[0]))[0]

  # Find highest p-value for comparison versus any of the best models.
  p_values = np.zeros(per_example_accuracies.shape[1])
  p_values[best_models] = 1
  for model_b in range(per_example_accuracies.shape[1]):
    if p_values[model_b] == 1:  # One of the best models.
      continue
    for model_a in best_models:
      p_value = modeljoust.compare(per_example_accuracies[:, model_a],
                                   per_example_accuracies[:, model_b],
                                   two_sided=TWO_SIDED.value)
      p_values[model_b] = max(p_values[model_b], p_value)

  # Determine the number of digits after the decimal point that should be shown.
  # We approximate the minimum accuracy delta that would be statistically
  # significant given the observed correct/incorrect agreement between the
  # best model and the closest model to the best. We ensure that enough digits
  # are shown that we would not report exactly the same accuracy for a model
  # with this minimum delta from the best as we do for the best model.
  agreements = np.mean(
      per_example_accuracies[:, best_models, None] ==
      per_example_accuracies[:, None, :], 0)
  if np.all(agreements == 1):
    max_agreement = 0.5  # Backup in case all predictions are identical.
  else:
    max_agreement = np.max(agreements[agreements != 1])
  detectable_delta = modeljoust.detectable_accuracy_delta(
      max_agreement, len(labels),
      two_sided=TWO_SIDED.value) * 100
  precision = int(np.ceil(-np.log10(detectable_delta)))

  # Show output.
  headers = ['model', METRIC.value, 'p_value']
  table = []
  order = np.argsort(mean_accuracies)
  accuracy_format = '5.' + str(precision) + 'f'
  for imodel in order:
    p_value = p_values[imodel]
    if p_value == 1:
      p_value = 'best'
    elif p_value > 0.01:
      p_value = f'{p_value:.2f}'
    elif p_value == 0:
      if METRIC.value == 'mean_per_class':
        p_value = f'<{1 / modeljoust.DEFAULT_NUM_PERMS:.1g}'
      else:
        p_value = '<1e-323'
    else:
      p_value = f'{p_value:.1g}'
    table.append([model_names[imodel], mean_accuracies[imodel] * 100,
                  p_value])
  print(tabulate(table, headers, tablefmt='github',
                 floatfmt=('', accuracy_format, '')))


if __name__ == '__main__':
  flags.mark_flag_as_required('labels')
  app.run(main)
