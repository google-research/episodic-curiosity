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

# This script runs as root when the VM instance starts.
# It launches a child script, after dropping the root user.

set -x

USER=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/user" -H "Metadata-Flavor: Google")
EPISODIC_CURIOSITY_DIR="/home/${USER}/episodic-curiosity"
PRETRAINED_R_NETS_PATH=$(curl "http://metadata.google.internal/computeMetadata/v1/instance/attributes/pretrained_r_nets_path" -H "Metadata-Flavor: Google")

# Note: during development, you could sync code from your local machine to a
# cloud bucket, sync it here from the bucket to the VM, and pip install it.

if [[ "${PRETRAINED_R_NETS_PATH}" ]]
then
  gsutil -m cp -r "${PRETRAINED_R_NETS_PATH}" "/home/${USER}"
fi

chmod a+x "${EPISODIC_CURIOSITY_DIR}/scripts/vm_start.sh"

# Launch vm_start under the given user.
su - "${USER}" -c "${EPISODIC_CURIOSITY_DIR}/scripts/vm_start.sh"
