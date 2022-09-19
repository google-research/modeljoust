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

"""Function for reading predictions file."""

import csv
from typing import Any, Dict, List


def read_predictions_file(path: str) -> Dict[Any, List[Any]]:
  """Reads a predictions/labels file.

  Predictions files are CSV files where the first column is the example
  filename and the remaining columns contain the top-k predictions in
  descending order. Labels files are formatted similarly except columns
  contain correct labels instead of predictions.

  Args:
    path: Path to predictions file.

  Returns:
    predictions: Dict of predictions.
  """
  predictions = {}
  with open(path, newline='') as f:
    reader = csv.reader(f)
    for row in reader:
      predictions[row[0]] = row[1:]
  return predictions
