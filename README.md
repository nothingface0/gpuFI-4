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

## Injection campaign execution and results

After setting up everything described on the previous section, the fault injection campaign can be easily executed
by simply running the campaign.sh script. The script eventually will go on a loop (until it reaches #RUNS cycles),
where each cycle will modify the framework’s new parameters at gpgpusim.config file before executing the application.

After completion of every batch of fault injections, a parser post-processes the output of the experiments one
by one and accumulates the results. The final results will be printed when all the batches have finished and
the script has quit. The parser classifies the fault effects of each experiment as Masked, Silent Data Corruption (SDC),
or Detected Unrecoverable Error (DUE).

- **Masked:** Faults in this category let the application run until the end and the result is identical to that of a fault-free execution.
- **Silent Data Corruption (SDC):** The behavior of an application with these types of faults is the same as with masked faults but the application’s result is incorrect. These faults are difficult to identify as they occur without any indication that a fault has been recorded (an abnormal event such as an exception, etc.).
- **Detected Unrecoverable Error (DUE):** In this case, an error is recorded and the application reaches an abnormal state without the ability to recover.

We additionally use the term “Performance” as a fault effect which is nothing but a Masked fault effect where the total cycles of the application are different from the fault-free execution.

## Available scripts

More detailed information about each available script can be found in this section.

## Known limitations

> [!IMPORTANT]
> TODO
