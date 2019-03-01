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

"""py3 script that creates all the VMs on GCP for episodic curiosity.

Right now, this launches the experiments to reproduce table 1 of
https://arxiv.org/pdf/1810.02274.pdf.

This script only depends on the standard py libraries on purpose, so that it can
be executed under any setup.

The gcloud command should be accessible (see:
https://cloud.google.com/sdk/gcloud/).

Invoke this script at the root of episodic-curiosity:
python3 scripts/launch_cloud_vms.py.

Tip: inspect the logs of the startup script on the VMs with:
sudo journalctl -u google-startup-scripts.service
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import subprocess
import concurrent.futures

parser = argparse.ArgumentParser()
parser.add_argument('--i_understand_launching_vms_is_expensive',
                    action='store_true',
                    help=('Each VM costs on the order of USD 30 per day based '
                          'on early 2019 GCP pricing. This script launches '
                          'many VMs at once, which can cost significant money. '
                          'Pass this flag to show that you understood this.'))
FLAGS = parser.parse_args()

# Information about your GCP account/project:
GCLOUD_PROJECT = 'FILL-ME'
SERVICE_ACCOUNT = 'FILL-ME'
ZONE = 'FILL-ME'
# GCP snapshot with episodic-curiosity (and its dependencies) installed as
# explained in README.md.
SOURCE_SNAPSHOT = 'FILL-ME'
# User to use on the VMs.
VM_USER = 'FILL-ME'
# Training logs and checkpoints will be synced to this Google Cloud bucket path.
# E.g. 'gs://my-episodic-curiosity-logs/training_logs'
SYNC_LOGS_TO_PATH = 'FILL-ME'

# Path on a Google Cloud bucket to the pre-trained R-networks. If empty,
# R-networks will be re-trained.
PRETRAINED_R_NETS_PATH = 'gs://episodic-curiosity/r_networks'


# Name templates for instances and disks.
NAME_TEMPLATE = 'ec-20190301-{method}-{scenario}-{run_number}'

# Scenarios to launch.
SCENARIOS = [
    'noreward',
    'norewardnofire',
    'sparse',
    'verysparse',
    'sparseplusdoors',
    'dense1',
    'dense2',
]

# Methods to launch.
METHODS = [
    'ppo',
    # This is the online version of episodic curiosity.
    'ppo_plus_eco',
    # This is the version of episodic curiosity where the R-network is trained
    # before the policy.
    'ppo_plus_ec',
    'ppo_plus_grid_oracle'
]

# Number of identical training jobs to launch for each scenario and method.
# Given the variance across runs, multiple of them are needed in order to get
# confidence in the results.
NUM_REPEATS = 10


def required_resources_for_method(method, uses_pretrained_r_net):
  """Returns the required resources for the given training method.

  Args:
    method: str, training method.
    uses_pretrained_r_net: bool, whether we use pre-trained r-net.

  Returns:
    Tuple (RAM (MBs), num CPUs, num GPUs)
  """
  if method == 'ppo_plus_eco':
    # We need to rent 2 GPUs, because with this amount of RAM, GCP won't allow
    # us to rent only one.
    return (105472, 16, 2)
  if method == 'ppo_plus_ec' and not uses_pretrained_r_net:
    return (52224, 12, 1)
  return (32768, 12, 1)


def launch_vm(vm_id, vm_metadata):
  """Creates and launches a VM on Google Cloud compute engine.

  Args:
    vm_id: str, unique ID of the vm.
    vm_metadata: Dict[str, str], metadata key/value pairs passed to the vm.
  """
  print('\nCreating disk and vm with ID:', vm_id)
  vm_metadata['vm_id'] = vm_id
  ram_mbs, num_cpus, num_gpus = required_resources_for_method(
      vm_metadata['method'],
      bool(vm_metadata['pretrained_r_nets_path']))

  create_disk_cmd = (
      'gcloud compute disks create '
      '"{disk_name}" --zone "{zone}" --source-snapshot "{source_snapshot}" '
      '--type "pd-standard" --project="{gcloud_project}" '
      '--size=200GB'.format(
          disk_name=vm_id,
          zone=ZONE,
          source_snapshot=SOURCE_SNAPSHOT,
          gcloud_project=GCLOUD_PROJECT,
      ))
  print('Calling', create_disk_cmd)
  # Don't fail if disk already exists.
  subprocess.call(create_disk_cmd, shell=True)

  create_instance_cmd = (
      'gcloud compute --project={gcloud_project} instances create '
      '{instance_name} --zone={zone} --machine-type={machine_type} '
      '--subnet=default --network-tier=PREMIUM --maintenance-policy=TERMINATE '
      '--service-account={service_account} '
      '--scopes=storage-full,compute-rw '
      '--accelerator=type=nvidia-tesla-p100,count={gpu_count} '
      '--disk=name={disk_name},device-name={disk_name},mode=rw,boot=yes,'
      'auto-delete=yes --restart-on-failure '
      '--metadata-from-file startup-script=./scripts/vm_drop_root.sh '
      '--metadata {vm_metadata} --async'.format(
          instance_name=vm_id,
          zone=ZONE,
          machine_type='custom-{num_cpus}-{ram_mbs}'.format(
              num_cpus=num_cpus, ram_mbs=ram_mbs),
          gpu_count=num_gpus,
          disk_name=vm_id,
          vm_metadata=(
              ','.join('{}={}'.format(k, v) for k, v in vm_metadata.items())),
          gcloud_project=GCLOUD_PROJECT,
          service_account=SERVICE_ACCOUNT,
      ))

  print('Calling', create_instance_cmd)
  subprocess.check_call(create_instance_cmd, shell=True)


def main():
  launch_args = []
  for method in METHODS:
    for scenario in SCENARIOS:
      for run_number in range(NUM_REPEATS):
        vm_id = NAME_TEMPLATE.format(
            method=method.replace('_', '-'),
            scenario=scenario.replace('_', '-'),
            run_number=run_number)
        launch_args.append((
            vm_id,
            {
                'method': method,
                'scenario': scenario,
                'run_number': str(run_number),
                'user': VM_USER,
                'pretrained_r_nets_path': PRETRAINED_R_NETS_PATH,
                'sync_logs_to_path': SYNC_LOGS_TO_PATH
            }))
  print('YOU ARE ABOUT TO START', len(launch_args), 'VMs on GCP.')
  if not FLAGS.i_understand_launching_vms_is_expensive:
    print('Please pass --i_understand_launching_vms_is_expensive to specify '
          'that you understood the cost implications of launching',
          len(launch_args), 'VMs')
    return
  # We use many threads in order to quickly start many instances.
  with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
    futures = []
    for args in launch_args:
      futures.append(executor.submit(launch_vm, *args))
    for f in futures:
      assert f.result() is None


if __name__ == '__main__':
  main()
