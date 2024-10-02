# gpuFI-4: A Microarchitecture-Level Framework for Assessing the Cross-Layer Resilience of Nvidia GPUs

gpuFI-4 is a detailed microarchitecture-level fault injection framework to assess the cross-layer vulnerability of hardware structures and entire Nvidia GPU chips for single and multiple bit faults, built on top of the state-of-the-art simulator [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution). The target hardware structures that gpuFI-4 can perform error injection campaigns are the register file, the shared memory, the L1 data, texture and instruction caches, as well as the L2 cache.

## Reference

If you use gpuFI-4 for your research, please cite:

> D. Sartzetakis, G. Papadimitriou, and D. Gizopoulos, “gpuFI-4: A Microarchitecture-Level Framework for Assessing the Cross-Layer Resilience of Nvidia GPUs", IEEE International Symposium on Performance Analysis of Systems and Software (ISPASS 2022), Singapore, May 22-24 2022.

The full ISPASS 2022 paper for gpuFI-4 can be found [here](http://cal.di.uoa.gr/wp-content/uploads/2022/04/gpuFI-4_ISPASS_2022.pdf).

## Building the simulator

gpuFI-4's requirements and building process are mostly identical to [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim_distribution).

If your purpose is to use gpuFI-4 to evaluate the fault effects of a CUDA application using PTXPLUS and not PTX then make sure that you are compiling GPGPU-Sim and the application with CUDA 4.2 or less as PTXPLUS currently only supports `sm_1x`.

### Prerequisites

To build and use gpuFI, you will be needing:

- A C++ compiler (we used the GNU toolchain, and tested with versions 9.5 up to 12.3).
- CUDA Toolkit version 4.2 (to be used with `CUDA_INSTALL_PATH`).
- CUDA Toolkit version 11.0 (to be used with `PTXAS_CUDA_INSTALL_PATH`).

Below, instructions on how to install both on a Linux distribution (Ubuntu versions 22.10 and 23.04 were tested) are given.

The CUDA Toolkit 4.2 can be installed by running:

```bash
curl -L http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/cudatoolkit_4.2.9_linux_64_ubuntu11.04.run -O
sudo chmod +x cudatoolkit_4.2.9_linux_64_ubuntu11.04.run
sudo sh cudatoolkit_4.2.9_linux_64_ubuntu11.04.run
```

Choose `/usr/local/cuda-4.2/cuda/` as the installation directory.

The CUDA Toolkit 11.0 can be installed by running:

```bash
curl -L http://developer.download.nvidia.com/compute/cuda/11.0.2/local_installers/cuda_11.0.2_450.51.05_linux.run -O
sudo sh cuda_11.0.2_450.51.05_linux.run
```

Choose `/usr/local/cuda-11.0/` as the installation directory.

### Step-by-step guide

To compile the simulator, make sure you've installed the [prerequisites](#prerequisites) first.

1. Clone or download the gpuFI source code from this repository and edit the `startup.sh` file inside the gpuFI-4 directory. Update the following values:

   - `PTXAS_CUDA_INSTALL_PATH`: Adapt this to your CUDA Toolkit 11.0 installation (e.g., `/usr/local/cuda-11.0`).
   - `CUDA_INSTALL_PATH` to point to your CUDA Toolkit 4.2 installation (e.g., `/usr/local/cuda-4.2/cuda`).

2. Then, open a terminal inside the same directory and run:

   ```bash
   source startup.sh
   make -j
   ```

   To compile without optimizations, in order to run with a debugger, run:

   ```bash
   source startup.sh debug
   make -j
   ```

## CUDA applications

For a CUDA application to be used with this simulator, the following requirements must be met:

- The application's source code should be available.
- The application should be slightly modified so that it runs a validation procedure before its exit, printing out if the validation passed or failed.

The second point is due to the fact that gpuFI-4 must have a way of knowing, at the end of each execution, if the application ran correctly or not.

### Modifying the applications

The applications to be used with the simulator should be slightly modified to run a comparison procedure of the results they produce from the GPU part of the execution.

Examples of how this can be achieved include:

- Generating a fault-free execution results file (once per application), by running the application without a fault injection, which can be used during each run of an injection campaign to compare the run's results against.
- Hardcoding the results of a reference execution inside the application, and running a validation against them at the end of each run.

In any case, a success (`Tests PASSED`) or a failure (`Tests FAILED`) message is expected to be found in the standard output of the application at the end of each run, in order to determine if data validation passed or not.

An example of how this was done can be found [here](https://github.com/nothingface0/gpu-rodinia/blob/gpgpu_sim_fi/cuda/bfs/bfs.cu) for the `bfs` Rodinia benchmark, with some sample reference input files [here](https://github.com/nothingface0/gpu-rodinia/tree/gpgpu_sim_fi/data/bfs).

### Building the applications

Considering the fact that gpuFI relies on running with the `gpgpu_ptx_convert_to_ptxplus` option enabled, the CUDA applications must be built (as of 2024/09) with CUDA Toolkit 4.2, to contain `sm_1x` SASS instructions. Since the CUDA Toolkit 4.2 requires gcc version 4.x, the most straightforward way to build the applications is inside a container.

> [!NOTE]
> For this example, we are using `docker` to compile a benchmark from the Rodinia suite. We are also assuming that the [source code](https://github.com/nothingface0/gpu-rodinia/tree/gpgpu_sim_fi) has been cloned into: `$HOME/Documents/workspace/gpu-rodinia`

First start the container:

```bash
sudo docker run --privileged \
-v $HOME/Documents/workspace/gpu-rodinia:/home/runner/gpu-rodinia \
aamodt/gpgpu-sim_regress:latest \
/bin/bash -c "tail -f /dev/null"
```

In a separate terminal, run the following to connect to a `sh` session inside the container:

```bash
sudo docker exec -it $(sudo docker ps | grep aamodt | awk '{print($1)}') \
/bin/bash -c "su -l runner"
```

From the terminal that opens inside the container, we can `cd` to the benchmark of our choosing, under the `gpu-rodinia/cuda` directory and issue `make`. The compiled CUDA executable will be available on the host operating system, alongside the CUDA application source code.

For example, to compile the `srad_v1` benchmark, from within the container run:

```bash
cd gpu-rodinia/cuda/srad/srad_v1
make clean
make
```

The compiled executable will then be found on the host OS in the path:
`$HOME/Documents/workspace/gpu-rodinia/cuda/srad/srad_v1/srad`

Once done, you can stop the running container with:

```bash
sudo docker kill $(sudo docker ps | grep aamodt | awk '{print($1)}')
```

## Injection campaigns

gpuFI-4’s operation is based on the concept of _injection campaigns_. A campaign is composed of multiple executions (_runs_) of the same combination of:

- a **GPU configuration**, i.e., a valid `gpgpusim.config` file,
- a (properly modified) **CUDA executable**, and
- a specific set of **executable arguments.**

Before each execution starts, a bit of the target memory type and an execution cycle are selected randomly. During execution, when the randomly selected execution cycle is reached, the randomized bit of the target memory is flipped (_bitflip_) and the execution is resumed. In the case that the target memory is an L1 cache, the shared memory, local memory or a register, an SM is also selected at random. For the L2 cache, such a selection is not required, as L2 cache is shared among SMs. When execution ends, or when it exceeds a predetermined timeout, the results are collected and stored in a `results.csv` file.

When gpuFI-4 selects a random SM, it limits the selection to those that are active during the executable’s execution. The same applies for the cycle, registers, constant memory, local memory and shared memory bits.

To know the exact limits of the selection, an _analysis_ of the executable has to be performed. This analysis is done in two steps:

1. Execution of the CUDA executable without fault injections. This creates a record of the total cycles that the kernels run for, as well as the simulation time.
2. Execution of the CUDA executable in “profiling mode”. This mode extracts information on SM, register, shared memory, constant and local memory usage. This information is known on the last cycle of simulation, which is the reason why step 1 is needed.

Once analysis is complete, a campaign can start, launching multiple executions, or “runs”, of the given executable with a randomized bitflip.

Injection campaigns are managed by the `gpufi_campaign.sh` script. This script manages the execution of a whole injection campaign on a specific combination of CUDA executable, arguments and GPU configuration file.

For each new run initiated by the script, a unique `run_id` is created. This is computed as the MD5 sum of:

1. the full contents of the `gpgpusim.config` file,
2. the full binary contents of the CUDA executable and
3. the full string of arguments passed to the executable.

The results of each run's bitflip are categorized as followed:

- **Masked:** Faults in this category let the application run until the end and the result is identical to that of a fault-free execution.
- **Silent Data Corruption (SDC):** The application reaches the end of execution without error, but the data verification step fails.
- **Detected Unrecoverable Error (DUE):** In this case, an error is recorded and the application reaches an abnormal state without the ability to recover (e.g., a crash).

We additionally use the term “Performance” to label faults, which is actually just a Masked fault, which has the extra effect of leading to different total execution cycles of the application, compared to the fault-free execution.

## Available scripts

More detailed information about each available script can be found in this section.

## Known limitations

> [!IMPORTANT]
> TODO
