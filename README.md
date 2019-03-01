## Episodic Curiosity Through Reachability

#### In ICLR 2019 [[Project Website](https://sites.google.com/corp/view/episodic-curiosity)][[Paper](https://arxiv.org/abs/1810.02274)]

[Nikolay Savinov¹](http://people.inf.ethz.ch/nsavinov/), [Anton Raichuk²](https://ai.google/research/people/AntonRaichuk), [Raphaël Marinier²](https://ai.google/research/people/105955), Damien Vincent², [Marc Pollefeys¹](https://www.inf.ethz.ch/personal/marc.pollefeys/), [Timothy Lillicrap³](http://contrastiveconvergence.net/~timothylillicrap/index.php), [Sylvain Gelly²](https://ai.google/research/people/SylvainGelly)<br/>
¹ETH Zurich, ²Google AI, ³DeepMind<br/>

Navigation out of curiosity                         | Locomotion out of curiosity
--------------------------------------------------- | ---------------------------
<img src="misc/navigation_github.gif" height="150"> | <img src="misc/ant_github.gif" height="150">

This is an implementation of our
[ICLR 2019 Episodic Curiosity Through Reachability](https://arxiv.org/abs/1810.02274).
If you use this work, please cite:

    @inproceedings{Savinov2019_EC,
        Author = {Savinov, Nikolay and Raichuk, Anton and Marinier, Rapha{\"e}l and Vincent, Damien and Pollefeys, Marc and Lillicrap, Timothy and Gelly, Sylvain},
        Title = {Episodic Curiosity through Reachability},
        Booktitle = {International Conference on Learning Representations ({ICLR})},
        Year = {2019}
    }

### Requirements

The code was tested on Linux only. The code assumes that the command "python"
invokes python 2.7. We recommend you use virtualenv:

```shell
sudo apt-get install python-pip
pip install virtualenv
python -m virtualenv episodic_curiosity_env
source episodic_curiosity_env/bin/activate
```

### Installation

Clone this repository:

```shell
git clone https://github.com/google-research/episodic-curiosity.git
cd episodic-curiosity
```

We require a modified version of
[DeepMind lab](https://github.com/deepmind/lab):

Clone DeepMind Lab:

```shell
git clone https://github.com/deepmind/lab
cd lab
```

Apply our patch to DeepMind Lab:

```shell
git checkout 7b851dcbf6171fa184bf8a25bf2c87fe6d3f5380
git checkout -b modified_dmlab
git apply ../third_party/dmlab/dmlab_min_goal_distance.patch
```

Install DMLab as a PIP module by following
[these instructions](https://github.com/deepmind/lab/tree/master/python/pip_package)

In a nutshell, once you've installed DMLab dependencies, you need to run:

```shell
bazel build -c opt python/pip_package:build_pip_package
./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
pip install /tmp/dmlab_pkg/DeepMind_Lab-1.0-py2-none-any.whl --force-reinstall
```

Finally, install episodic curiosity and its pip dependencies:

```shell
cd episodic-curiosity
pip install -e .
```

### Resource requirements for training

| Environment | Training method | Required GPU         | Recommended RAM            |
| ----------- | ---------- | -------------------- | -------------------------- |
| DMLab       | PPO        | No                   | 32GBs                      |
| DMLab       | PPO + Grid Oracle | No                   | 32GBs                      |
| DMLab       | PPO + EC using already trained R-networks   | No                   | 32GBs                      |
| DMLab       | PPO + EC with R-network training   | Yes, otherwise, training is slower by >20x.<br>Required GPU RAM: 5GBs      | 50GBs<br>Tip: reduce  `dataset_buffer_size` for  using less RAM at the expense of policy performance.   |
| DMLab       | PPO + ECO  | Yes, otherwise, raining is slower by >20x.<br>Required GPU RAM: 5GBs     | 80GBs<br>Tip: reduce `observation_history_size` for using less RAM, at the expense of policy performance      |


## Trained models

Trained R-networks can be found in the `episodic-curiosity` Google cloud bucket.
You can access them via the
[web interface](https://console.cloud.google.com/storage/browser/episodic-curiosity/r_networks),
or copy them with the `gsutil` command from the
[Google Cloud SDK](https://cloud.google.com/sdk):

```shell
gsutil -m cp gs://episodic-curiosity/r_networks .
```

We plan to also release the pre-trained policies soon.

## Training

### On a single machine

[scripts/launcher_script.py](https://github.com/google-research/episodic-curiosity/blob/master/scripts/launcher_script.py)
is the main entry point to reproduce the results of Table 1 in the
[paper](https://arxiv.org/abs/1810.02274). For instance, the following command
line launches training of the *PPO + EC* method on the *Sparse+Doors* scenario:

```sh
python episodic_curiosity/scripts/launcher_script.py --workdir=/tmp/ec_workdir --method=ppo_plus_ec --scenarios=sparseplusdoors
```

Main flags:

| Flag | Descriptions |
| :----------- | :--------- |
| --method | Solving method to use, corresponds to the rows in table 1 of the [paper](https://arxiv.org/abs/1810.02274). Possible values: `ppo, ppo_plus_ec, ppo_plus_eco, ppo_plus_grid_oracle` |
| --scenario | Scenario to launch. Corresponds to the columns in table 1 of the [paper](https://arxiv.org/abs/1810.02274). Possible values: `noreward, norewardnofire, sparse, verysparse, sparseplusdoors, dense1, dense2` |
| --workdir | Directory where logs and checkpoints will be stored.  |
| --run_number | Run number of the current run. This is used to create an appropriate subdir in workdir.  |
| --r_networks_path | Only meaningful for the `ppo_plus_ec` method. Path to the root dir for pre-trained r networks.  If specified, we train the policy using those pre-trained r networks. If not specified, we first generate the R network training data, train the R network and then train the policy. |


Training takes a couple of days. We used CPUs with 16 hyper-threads, but smaller
CPUs should do.

Under the hood,
[launcher_script.py](https://github.com/google-research/episodic-curiosity/blob/master/scripts/launcher_script.py)
launches
[train_policy.py](https://github.com/google-research/episodic-curiosity/blob/master/episodic_curiosity/train_policy.py)
with the right hyperparameters. For the method `ppo_plus_ec`, it first launches
[generate_r_training_data.py](https://github.com/google-research/episodic-curiosity/blob/master/episodic_curiosity/generate_r_training_data.py)
to accumulate training data for the R-network using a random policy, then
launches
[train_r.py](https://github.com/google-research/episodic-curiosity/blob/master/episodic_curiosity/train_r.py)
to train the R-network, and finally
[train_policy.py](https://github.com/google-research/episodic-curiosity/blob/master/episodic_curiosity/train_policy.py)
for the policy. In the method `ppo_plus_eco`, all this happens online as part of
the policy training.

### On Google Cloud

First, make sure you have the [Google Cloud SDK](https://cloud.google.com/sdk)
installed.

[scripts/launch_cloud_vms.py](https://github.com/google-research/episodic-curiosity/blob/master/scripts/launch_cloud_vms.py)
is the main entry point. Edit the script and replace the `FILL-ME`s with the
details of your GCP project. In particular, you will need to point it to a GCP
disk snapshot with the installed dependencies as described in the
[Installation](#Installation) section.

IMPORTANT: By default the script reproduces all results in table 1 and launches
~300 VMs on cloud with GPUs (7 scenarios x 4 methods x 10 runs). The cost of
running all those VMs is very significant: on the order of USD 30 **per day**
**per VM** based on early 2019 GCP pricing. Pass
`--i_understand_launching_vms_is_expensive` to
[scripts/launch_cloud_vms.py](https://github.com/google-research/episodic-curiosity/blob/master/scripts/launch_cloud_vms.py)
to indicate that you understood that.

Under the hood, `launch_cloud_vms.py` launches one VM for each (scenario,
method, run_number) tuple. The VMs use startup scripts to launch training, and
retrieve the parameters of the run through
[Instance Metadata](https://cloud.google.com/compute/docs/storing-retrieving-metadata).

TIP: Use `sudo journalctl -u google-startup-scripts.service` to see the logs of
the startup script.

### Training logs

Each training job stores logs and checkpoints in a workdir. The workdir is
organized as follows:

| File or Directory                          | Description                     |
| :----------------------------------------- | :------------------------------ |
| `r_training_data/{R_TRAINING,VALIDATION}/` | TF Records with data generated from a random policy for R-network training. Only for method `ppo_plus_ec` without supplying pre-trained R-networks. |
| `r_networks/`                              | Keras checkpoints of trained R-networks. Only for method `ppo_plus_ec` without supplying pre-trained R-networks. |
| `reward_{train,valid,test}.csv`            | CSV files with {train,valid,test} rewards, tracking the performance of the policy at multiple training steps. |
| `checkpoints/`                             | Checkpoints of the policy.      |
| `log.txt`, `progress.csv`                  | Training logs and CSV from OpenAI's PPO2 code. |

On cloud, the workdir of each job will be synced to a cloud bucket directory of
the form `<cloud_bucket_root>/<vm_id>/<method>/<scenario>/run_number_<d>/`.

We provide a
[colab](https://github.com/google-research/episodic-curiosity/blob/master/colab/plot_training_graphs.ipynb)
to plot graphs during training of the policies, using data from the
`reward_{train,valid,test}.csv` files.

### Known limitations

-   As of 2019/02/20, `ppo_plus_eco` method is not robust to restarts, because
    the R-network trained online is not checkpointed.
-   This repo only covers training on Deepmind Lab. We are also considering
    releasing the code for training on Mujoco in the future.

