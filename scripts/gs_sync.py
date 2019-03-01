# coding=utf-8
# Copyright 2019 Google LLC.
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

"""Script that periodically syncs a dir to a cloud storage bucket.

We only use standard python deps so that this script can be executed in any
setup.
This script repeatedly calls the 'gsutil rsync' command.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--workdir', help='Source to sync to the cloud bucket')
parser.add_argument(
    '--sync_to_cloud_bucket',
    help='Cloud bucket (format gs://bucket_name) to sync the workdir to')
FLAGS = parser.parse_args()


def sync_to_cloud_bucket():
  """Repeatedly syncs a path to a cloud bucket using gsutil."""
  sync_cmd = (
      'gsutil -m rsync -r {src} {dst}'.format(
          src=os.path.expanduser(FLAGS.workdir),
          dst=os.path.join(
              FLAGS.sync_to_cloud_bucket, os.path.basename(FLAGS.workdir))))
  while True:
    print('Syncing to cloud bucket:', sync_cmd)
    # We don't stop on failure, it can be transcient issue, a subsequent run
    # might work.
    subprocess.call(sync_cmd, shell=True)
    time.sleep(60)


if __name__ == '__main__':
  sync_to_cloud_bucket()
