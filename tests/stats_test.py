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

"""Tests for stats."""

from absl.testing import absltest
import modeljoust
import numpy as np

# Per-example accuracies for which we have computed a p-value using
# https://www.omnicalculator.com/statistics/mcnemars-test
MODEL_A_CORRECT = np.array(
    [True] * 23 + [True] * 78 + [False] * 54 + [False] * 32)
MODEL_B_CORRECT = np.array(
    [True] * 23 + [False] * 78 + [True] * 54 + [False] * 32)
MODEL_C_CORRECT = np.array(
    [True] * 30 + [False] * 71 + [True] * 54 + [False] * 32)
MODEL_A_VS_B_P_VALUE = 0.04488539
MODEL_A_VS_C_P_VALUE = 0.15212087


class StatsTest(absltest.TestCase):

  def test_sign_test_two_sided(self):
    self.assertAlmostEqual(
        modeljoust.sign_test(MODEL_A_CORRECT, MODEL_B_CORRECT),
        MODEL_A_VS_B_P_VALUE)

  def test_sign_test_one_sided(self):
    self.assertAlmostEqual(
        modeljoust.sign_test(
            MODEL_A_CORRECT, MODEL_B_CORRECT, two_sided=False),
        MODEL_A_VS_B_P_VALUE / 2)

  def test_permutation_test_two_sided(self):
    self.assertAlmostEqual(
        modeljoust.permutation_test(
            MODEL_A_CORRECT, MODEL_B_CORRECT, num_perms=1_000_000),
        MODEL_A_VS_B_P_VALUE,
        places=3)

  def test_permutation_test_one_sided(self):
    self.assertAlmostEqual(
        modeljoust.permutation_test(
            MODEL_A_CORRECT, MODEL_B_CORRECT, num_perms=1_000_000,
            two_sided=False),
        MODEL_A_VS_B_P_VALUE / 2,
        places=3)

  def test_detectable_accuracy_delta(self):
    delta = modeljoust.detectable_accuracy_delta(0.1, 100_000, 0.05)
    a_correct = np.concatenate((
        np.zeros((45_000,), dtype=bool),
        np.ones((45_000,), dtype=bool),
        np.ones((10_000,), dtype=bool)))
    b_correct = np.concatenate((
        np.ones((45_000,), dtype=bool),
        np.zeros((45_000,), dtype=bool),
        np.ones((10_000,), dtype=bool)))
    # Fudge factor of 1 needed because normal approximation is imperfect.
    a_correct[:int(np.ceil(delta / 2 * 100_000)) + 1] = True
    b_correct[:int(np.ceil(delta / 2 * 100_000)) + 1] = False
    self.assertLess(modeljoust.sign_test(a_correct, b_correct), 0.05)

  def test_compare_mcnemar(self):
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT, MODEL_A_CORRECT),
                           1.0)
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT, MODEL_B_CORRECT),
                           MODEL_A_VS_B_P_VALUE)
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT, MODEL_C_CORRECT),
                           MODEL_A_VS_C_P_VALUE)

  def test_compare_permutation_test(self):
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT * 2,
                                              MODEL_A_CORRECT * 2,
                                              num_perms=1_000_000),
                           1.0)
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT * 2,
                                              MODEL_B_CORRECT * 2,
                                              num_perms=1_000_000),
                           MODEL_A_VS_B_P_VALUE, places=3)
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT * 2,
                                              MODEL_C_CORRECT * 2,
                                              num_perms=1_000_000),
                           MODEL_A_VS_C_P_VALUE, places=3)

  def test_compare_mcnemar_one_sided(self):
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT, MODEL_A_CORRECT,
                                              two_sided=False),
                           1.0)
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT, MODEL_B_CORRECT,
                                              two_sided=False),
                           MODEL_A_VS_B_P_VALUE / 2)
    self.assertAlmostEqual(modeljoust.compare(MODEL_A_CORRECT, MODEL_C_CORRECT,
                                              two_sided=False),
                           MODEL_A_VS_C_P_VALUE / 2)

  def test_per_example_accuracies_matrices(self):
    rng = np.random.default_rng(1337)
    correct_labels = rng.integers(10, size=(200,))
    results_a = np.full((200, 3), 100)
    results_a[:100, 0] = correct_labels[:100]
    results_b = np.full((200, 3), 100)
    results_b[:100, 1] = correct_labels[:100]
    results_b[100:, 2] = correct_labels[100:]
    results_c = np.full((200, 3), 100)
    results_c[100:, 2] = correct_labels[100:200]
    np.testing.assert_array_equal(
        modeljoust.get_per_example_accuracies(
            [results_a, results_b, results_c], correct_labels, metric='top1'),
        np.array([[True] * 100 + [False] * 100,
                  [False] * 200, [False] * 200]).T)
    np.testing.assert_array_equal(
        modeljoust.get_per_example_accuracies(
            [results_a, results_b, results_c], correct_labels, metric='top2'),
        np.array([[True] * 100 + [False] * 100,
                  [True] * 100 + [False] * 100, [False] * 200]).T)
    np.testing.assert_array_equal(
        modeljoust.get_per_example_accuracies(
            [results_a, results_b, results_c], correct_labels, metric='top3'),
        np.array([[True] * 100 + [False] * 100,
                  [True] * 200,
                  [False] * 100 + [True] * 100]).T)

  def test_per_example_accuracies_dicts(self):
    rng = np.random.default_rng(1337)
    example_ids = np.arange(1_000_000, 1_000_200)
    correct_labels = dict(zip(example_ids, rng.integers(10, size=(200,))))
    results_base = {k: [100, 100, 100] for k in example_ids}
    results_a = results_base.copy()
    results_a.update({k: [correct_labels[k], 100, 100]
                      for k in example_ids[:100]})
    results_b = results_base.copy()
    results_b.update({k: [100, correct_labels[k], 100]
                      for k in example_ids[:100]})
    results_b.update({k: [100, 100, correct_labels[k]]
                      for k in example_ids[100:]})
    results_c = results_base.copy()
    results_c.update({k: [100, 100, correct_labels[k]]
                      for k in example_ids[100:]})
    np.testing.assert_array_equal(
        modeljoust.get_per_example_accuracies(
            [results_a, results_b, results_c], correct_labels, metric='top1'),
        np.array([[True] * 100 + [False] * 100,
                  [False] * 200, [False] * 200]).T)
    np.testing.assert_array_equal(
        modeljoust.get_per_example_accuracies(
            [results_a, results_b, results_c], correct_labels, metric='top2'),
        np.array([[True] * 100 + [False] * 100,
                  [True] * 100 + [False] * 100, [False] * 200]).T)
    np.testing.assert_array_equal(
        modeljoust.get_per_example_accuracies(
            [results_a, results_b, results_c], correct_labels, metric='top3'),
        np.array([[True] * 100 + [False] * 100,
                  [True] * 200,
                  [False] * 100 + [True] * 100]).T)

  def test_per_example_accuracies_per_class(self):
    rng = np.random.default_rng(1337)
    num_classes = 10
    correct_labels = rng.integers(num_classes, size=(200,))
    num_per_class = np.bincount(correct_labels, minlength=num_classes)
    results_a = np.full((200, 3), 100)
    results_a[:100, 0] = correct_labels[:100]
    results_b = np.full((200, 3), 100)
    results_b[:100, 1] = correct_labels[:100]
    results_b[100:, 2] = correct_labels[100:]
    results_c = np.full((200, 3), 100)
    results_c[100:, 2] = correct_labels[100:200]
    correct = np.array([[True] * 100 + [False] * 100,
                        [False] * 200,
                        [False] * 200]).T
    per_example_accuracies = modeljoust.get_per_example_accuracies(
        [results_a, results_b, results_c], correct_labels,
        metric='mean_per_class')
    np.testing.assert_allclose(
        np.mean(per_example_accuracies, 0),
        np.sum(correct / num_per_class[correct_labels, None], 0) /
        len(num_per_class))
    np.testing.assert_allclose(
        per_example_accuracies,
        correct / (num_per_class[correct_labels, None] * len(num_per_class)) *
        per_example_accuracies.shape[0])


if __name__ == '__main__':
  absltest.main()
