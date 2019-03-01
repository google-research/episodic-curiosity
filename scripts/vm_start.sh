#! /bin/bash

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

# This script runs when the VM instance starts. It launches episodic training
# using launcher_script.py, and using the per-instance metadata (preferably set
# using launch_cloud_vms.py).

set -x

whoami

cd

VM_ID=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/vm_id" -H "Metadata-Flavor: Google")
METHOD=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/method" -H "Metadata-Flavor: Google")
SCENARIO=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/scenario" -H "Metadata-Flavor: Google")
RUN_NUMBER=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/run_number" -H "Metadata-Flavor: Google")
PRETRAINED_R_NETS_PATH=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/pretrained_r_nets_path" -H "Metadata-Flavor: Google")
SYNC_LOGS_TO_PATH=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/sync_logs_to_path" -H "Metadata-Flavor: Google")
HOMEDIR=$(pwd)

WORKDIR="${HOMEDIR}/${VM_ID}"

EPISODIC_CURIOSITY_DIR="${HOMEDIR}/episodic-curiosity"

mkdir -p "${WORKDIR}"

# This must happen before we activate the virtual env, otherwise, gsutil does not work.
python "${EPISODIC_CURIOSITY_DIR}/scripts/gs_sync.py" --workdir="${WORKDIR}" --sync_to_cloud_bucket="${SYNC_LOGS_TO_PATH}" &

source episodic_curiosity_env/bin/activate

cd "${EPISODIC_CURIOSITY_DIR}"

if [[ "${PRETRAINED_R_NETS_PATH}" ]]
then
  BASENAME=$(basename "${PRETRAINED_R_NETS_PATH}")
  R_NETWORKS_PATH_FLAG="--r_networks_path=${HOMEDIR}/${BASENAME}"
else
  R_NETWORKS_PATH_FLAG=""
fi

python "${EPISODIC_CURIOSITY_DIR}/scripts/launcher_script.py" --workdir="${WORKDIR}" --method="${METHOD}" --scenario="${SCENARIO}" --run_number="${RUN_NUMBER}" ${R_NETWORKS_PATH_FLAG}
