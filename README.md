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

### `gpufi_analyze_executable.sh`

This script is responsible for executing a series of analyses for each unique combination of a CUDA executable, a set of arguments and a specific GPU configuration. Its goal is to create a series of files which contain information that gpuFI will need in order to start a campaign.

This script executes the CUDA executable twice: once with the `gpufi_profile` option of `gpgpusim.config` set to `3` (no fault injection) and a second time with the same option set to 1 (profiling mode). During this process, a series of directories and files are created in the same directory where the CUDA executable is:

- A main directory named `.gpufi`, where all the analysis files are stored.
  - A directory for the specific GPU configuration used, e.g., `SM6_TITANX`. A copy of the `gpgpusim.config` file is also stored here.
    - A directory named using the specific arguments passed to the executable, after sanitization.
      - One directory for each kernel invoked by the executable, in which two files may be found: `cycles.txt` (a file of all the simulated cycles during which the kernel is active) and `kernel_analysis.sh` (which lists the ids of the SMs that the kernel was assigned to, as well as the registers and total sizes of several types of memories that the kernel used).
      - A file named executable_analysis.sh, which contains executable-wide statistics, including the total simulated cycles that the executable runs for and the maximum timeout value (in seconds), that the campaign should wait for before stopping execution.
      - A file named `.analysis_complete` which signals the fact that the analysis script has been executed for this specific combination of GPU configuration, executable and set of arguments.
      - A directory named results will be created here once a campaign is started, storing not only a CSV file with all the results of each run, but also the g-zipped configuration file of each run, for future reference.

As an example, to analyze the Rodinia NW benchmark with arguments `288 10` and for the `SM6_TITANX` configuration, one should run:

```bash
bash gpufi_analyze_executable.sh CUDA_EXECUTABLE_PATH=$HOME/Documents/workspace/gpu-rodinia/cuda/nw/needle CUDA_EXECUTABLE_ARGS="288 10" GPU_ID=SM6_TITANX
```

A detailed list of arguments accepted by the script can be seen below:

| Argument                | Required | Functionality                                                                                                                                                 | Default value |
| ----------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `CUDA_EXECUTABLE_PATH`  | Yes      | Specifies the path to the CUDA executable to run.                                                                                                             | None          |
| `CUDA_EXECUTABLE_ARGS`  | Yes      | Specifies the arguments for the CUDA executable to run, enclosed in double quotes.                                                                            | None          |
| `GPU_ID`                | Yes      | Specifies the base GPU configuration file to use. It must be the name of one of the folders under the `configs/tested-configs` directory, e.g., `SM6_TITANX`. | None          |
| `do_execute_executable` | No       | Enables running the actual analysis or not. Mostly for debugging purposes.                                                                                    | `1`           |

### `gpufi_campaign.sh`

This script manages the execution of a whole injection campaign on a specific combination of CUDA executable, arguments and GPU configuration file. It covers the functionality
needed for injection campaigns.

Running the campaign script requires that the aforementioned combination has been analyzed (see [`gpufi_analyze_executable.sh`](#gpufi_analyze_executablesh)).

For each new run initiated by the script, a unique run_id is created. This is computed as
the MD5 sum of:

1. the full contents of the `gpgpusim.config` file,
2. the full binary contents of the CUDA executable and
3. the full string of arguments passed to the executable.

As an example, for executing an injection campaign of 2000 runs for the Rodinia NW benchmark, with arguments `288 10`, using the `SM6_TITANX` configuration one should run:

```bash
bash gpufi_campaign.sh CUDA_EXECUTABLE_PATH=$HOME/Documents/workspace/gpu-rodinia/cuda/nw/needle CUDA_EXECUTABLE_ARGS="288 10" GPU_ID=SM6_TITANX NUM_RUNS=2000
```

A list of arguments accepted by the script can be found below.

| Argument               | Required | Functionality                                                                                                                                                                                                                | Default value                        |
| ---------------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `CUDA_EXECUTABLE_PATH` | Yes      | Specifies the path to the CUDA executable to run.                                                                                                                                                                            | None                                 |
| `CUDA_EXECUTABLE_ARGS` | Yes      | Specifies the arguments for the CUDA executable to run, enclosed in double quotes (`"`).                                                                                                                                     | None                                 |
| `GPU_ID`               | Yes      | Specifies the base GPU configuration file to use. Must be the name of one of the folders under `configs/tested-configs`, e.g., `SM6_TITANX`.                                                                                 | None                                 |
| `NUM_RUNS`             | Yes      | Specifies the number of injections to run.                                                                                                                                                                                   | None                                 |
| `COMPONENTS_TO_FLIP`   | No       | Specifies the GPU components to target during the campaign, semicolon-separated if many are required. `0`: Register file, `1`: Local mem, `2`: Shared mem, `3`: L1D, `4`: L1C (not implemented), `5`: L1T, `6`: L2, `7`: L1I | 7                                    |
| `_NUM_AVAILABLE_CORES` | No       | Specifies the number of CPU cores to use for the simulations.                                                                                                                                                                | `$(nproc)` (all available CPU cores) |
| `DELETE_LOGS`          | No       | Delete temporary execution log files.                                                                                                                                                                                        | `1`                                  |
| `KERNEL_INDICES`       | No       | Specifies the id of the kernel to inject. (Not implemented yet)                                                                                                                                                              | `0` (all kernels)                    |

### `gpufi_replay_run.sh`

A bash script which allows the replay of a specific injection run which is present in the results.csv file, for a specific CUDA executable and its arguments, and GPU configuration. The `RUN_ID` provided must be a valid run id, and the corresponding `<RUN_ID>.tar.gz` file must be located in the configs subdirectory of the `.gpufi` directory.

A detailed list of arguments accepted by this script can be found below:

| Argument               | Required | Functionality                                                                                                                                | Default value    |
| ---------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `CUDA_EXECUTABLE_PATH` | Yes      | Specifies the path to the CUDA executable to run.                                                                                            | None             |
| `CUDA_EXECUTABLE_ARGS` | Yes      | Specifies the arguments for the CUDA executable to run, enclosed in double quotes (`"`).                                                     | None             |
| `GPU_ID`               | Yes      | Specifies the base GPU configuration file to use. Must be the name of one of the folders under `configs/tested-configs`, e.g., `SM6_TITANX`. | None             |
| `RUN_ID`               | Yes      | The unique 32-digit MD5sum of the run to replay.                                                                                             | None             |
| `_OUTPUT_LOG`          | No       | The filepath where the log of the replay will be stored.                                                                                     | `./<RUN_ID>.log` |

### Helper: `gpufi_calculate_cache_sizes.sh`

A helper script used to calculate all cache sizes configured in a `gpgpusim.config` file. It also exports the following bash environment variables for each type of cache, also used by `gpufi_campaign.sh`:

- `x_SIZE_BITS`
- `x_TAG_BITS`
- `x_ASSOC`
- `x_NUM_SETS`
- `x_BYTES_PER_LINE`
- `x_BITS_FOR_BYTE_OFFSET`

Where `x` is replaced with `L1D`, `L1C`, `L1T`, `L1I`, `L2`, meaning that a total of 30 environmental variables are exported by this script.

The arguments accepted by the script can be found below:

| Argument      | Required | Functionality                                     | Default value       |
| ------------- | -------- | ------------------------------------------------- | ------------------- |
| `CONFIG_FILE` | Yes      | Specifies the path to the `gpgpusim.config` file. | `./gpgpusim.config` |

### Helper: `gpufi_inject_cuda_executable.sh`

This is the script invoked by gpuFI to replace a specific byte sequence with another one in a CUDA executable, used during L1I cache injections.

This is the only script which does not accept its arguments in `KEY=VALUE` format, but as **positional arguments** instead.

The arguments accepted by the script can be found below:

| Argument Position | Functionality                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------------------- |
| 0                 | Specifies the full path of the CUDA executable to inject.                                         |
| 1                 | Specifies the full path to place the injected executable to.                                      |
| 2                 | The byte sequence (i.e., the instruction) in hex format (without `0x`) to look for and replace.   |
| 3                 | The byte sequence (i.e., the instruction) in hex format (without `0x`) to replace with.           |
| 4                 | The (mangled) kernel name to limit the search and replacement to. **NOTE:** Not fully functional. |

## Full campaign execution transcript

As a reference, the full procedure to run a campaign of 2000 L1I injections for the Rodinia NW benchmark, with arguments `288 10` for the Titan X (Pascal) GPU (`SM6_TITANX`) is described here. It assumes that you have cloned the source code of gpuFI in `$HOME/Documents/workspace/gpu-rodinia` and the modified rodinia benchmarks in `$HOME/Documents/workspace/gpuFI-4` and that you have CUDA 4.2 installed in `/usr/local/cuda-4.2/cuda/` and CUDA 11.0 installed in `/usr/local/cuda-11.0/`.

```bash
# Step 1, in a new terminal.
# Starts a docker container where the benchmarks will be compiled in.
# The output directory is mounted from the host OS, so that the binary will be
# readily available to the host OS.
sudo docker run --privileged \
-v $HOME/Documents/workspace/gpu-rodinia:/home/runner/gpu-rodinia \
aamodt/gpgpu-sim_regress:latest \
/bin/bash -c "tail -f /dev/null"
```

```bash
# Step 2, in a second terminal.
# Connect to the docker container, and build the benchmark.
sudo docker exec -it $(sudo docker ps | grep aamodt | awk '{print($1)}') \
/bin/bash -c "su -l runner"
cd gpu-rodinia/cuda/nw
make clean
make
```

```bash
# Step 3, in a third terminal.
# In the host OS, build the simulator and start the campaign.
cd $HOME/Documents/workspace/gpuFI-4
source startup.sh
make -j
# Run analysis
bash gpufi_analyze_executable.sh CUDA_EXECUTABLE_PATH=$HOME/Documents/
workspace/gpu-rodinia/cuda/nw/needle CUDA_EXECUTABLE_ARGS="288 10" GPU_ID=
SM6_TITANX
# Start the campaign
bash gpufi_campaign.sh CUDA_EXECUTABLE_PATH=$HOME/Documents/workspace/gpu-
rodinia/cuda/nw/needle CUDA_EXECUTABLE_ARGS="288 10" GPU_ID=SM6_TITANX
KERNEL_INDICES=0 DELETE_LOGS=0 NUM_RUNS=2000
```

## Known limitations

- No support for SM microarchitectures >= 20 (Fermi and above), as GPGPU-Sim does not support them for PTXPlus simulation.
- Some rare corner cases of modified instructions are not covered by the modified simulator, and some may lead to simulator crashes, e.g., cases where the injected instruction references memory spaces that have not been initialized by the simulator might not behave properly.
- No support for sectored L1I cache configuration.
- No support for L2 cache bitflips which should propagate to L1I.
- It was observed that some benchmarks, namely LUD, SRAD_v1 and SRAD_v2, when simulated with the QV100 configuration, might lead to different execution cycles for no apparent reason. To be precise, some of the randomized campaign configurations (roughly 1 out of 5000 runs), which do not lead to either a tag bitflip or a data bitflip on valid cache lines, might execute in a different number of total cycles than expected. Re-running the same configuration multiple times, we could not reproduce the problem. The workaround was to rerun the same run configuration, leading to the correct results.
