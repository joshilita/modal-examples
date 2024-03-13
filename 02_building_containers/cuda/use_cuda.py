# # Using CUDA with Modal
#
# CUDA is not one library, but a stack with multiple layers:
#
# - The NVIDIA CUDA Drivers
#   - kernel level components, like nvidia.ko
#   - the user level CUDA Driver API, libcuda.so
# - The NVIDIA CUDA Toolkit
#   - the CUDA Runtime API (libcudart.so)
#   - the NVIDIA CUDA compiler (nvcc)
#   - and more goodies
#
# Most folks working in ML don't use these things directly,
# and instead use them via a higher-level library like TensorFlow or PyTorch.
#
# But when configuring environments to run ML applications,
# even using those higher-level libraries, issues with the CUDA stack
# still arise and cause major headaches.
#
# In this tutorial, we'll tour the NVIDIA CUDA stack layer by layer,
# showing how to use it (and break it!) with Modal.

from modal import Image, Mount, Stub, gpu

stub = Stub()

base_image = Image.debian_slim()

GPU_CONFIG = gpu.T4()

# All Modal containers with a GPU attached have the NVIDIA CUDA drivers
# installed and the CUDA Driver API available.


@stub.function(gpu=GPU_CONFIG, image=base_image)
def nvidia_smi():
    import subprocess
    import xml.etree.ElementTree as ET

    # nvidia-smi runs, see the output in your terminal
    assert not subprocess.run(["nvidia-smi"]).returncode

    # let's check the output programmatically
    xml_output = subprocess.run(
        ["nvidia-smi", "-q", "-u", "-x"], capture_output=True
    ).stdout

    root = ET.fromstring(xml_output)

    assert root.find("driver_version").text.split(".")[0] == "535"
    assert root.find("cuda_version").text.split(".") == ["12", "2"]


# Even if we remove the CUDA Driver API, the NVIDIA drivers are still present.
# This is mostly a curiosity, but it underscores the difference between
# ther raw device drivers and the user-level C API to those drivers.


@stub.function(gpu=GPU_CONFIG, image=base_image)
def remove_libcuda(verbose: bool = True):
    import os
    import subprocess
    import xml.etree.ElementTree as ET
    from pathlib import Path

    root = Path("/")
    shared_user_level_dir = root / "usr"
    shared_library_dir = shared_user_level_dir / "lib"
    shared_x86_dir = shared_library_dir / "x86_64-linux-gnu"

    # remove libnvidia-ml.so and related files
    for libcuda_file in shared_x86_dir.glob("libnvidia-ml*"):
        if verbose:
            print("removing", libcuda_file)
        os.remove(libcuda_file)
    if verbose:
        print()  # empty line

    xml_output = subprocess.run(
        ["nvidia-smi", "-q", "-u", "-x"], capture_output=True, check=False
    ).stdout
    if verbose:
        print("nvidia-smi still runs!")

    root = ET.fromstring(xml_output)
    assert root.find("driver_version").text.split(".")[0] == "535"
    if verbose:
        print("because the NVIDIA drivers are still present")
    assert root.find("cuda_version").text.lower() == "not found"
    if verbose:
        print("but the CUDA Driver API is gone:", "\n")
        subprocess.run(["nvidia-smi"])


# Even with nothing but the drivers and their API,
# we can still run packaged CUDA programs.
#
# The function below does this in a silly, non-standard way,
# just to get the point across:
# we send the compiled CUDA program as a bunch of bytes, then run it.


@stub.function(gpu=GPU_CONFIG, image=base_image)
def raw_cuda(prog: bytes, with_libcuda: bool = True):
    import os
    import subprocess

    if not with_libcuda:
        remove_libcuda.local(verbose=False)

    with open("./prog", "wb") as f:
        f.write(prog)

    os.chmod("./prog", 0o755)
    # TODO: switch to real error-handling in the C code, so this raises an exception
    subprocess.run(["./prog"])


# But to use that, we'll need a compiled CUDA program to run.
# So let's install the NVIDIA CUDA compiler (nvcc) and the CUDA Toolkit.
# We'll do this on a new image, to underscore that we don't need to install
# these dependencies just to run CUDA programs.


arch = "x86_64"  # instruction set architecture for the CPU, all Modal machines are x86_64
distro = "debian11"  # the distribution and version number of our OS (GNU/Linux)
cuda_keyring_url = f"https://developer.download.nvidia.com/compute/cuda/repos/{distro}/{arch}/cuda-keyring_1.1-1_all.deb"


cudatoolkit_image = (
    base_image.apt_install("wget")
    .run_commands(
        [  # we need to get hold of NVIDIA's CUDA keyring to verify the installation
            f"wget {cuda_keyring_url}",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
        ]  # otherwise we can't be sure the binaries are from NVIDIA
    )
    .apt_install(
        "cuda-compiler-12-1"
    )  # MUST BE <= 12.2! TODO: write up fwd/bwd compatibility
    .env({"PATH": "/usr/local/cuda/bin:$PATH"})
)

# Now we can use nvcc to compile a CUDA program.
#
# We've included a simple CUDA program with this example.
# It's a translation of John Carmack's famous fast inverse square root algorithm
# from Quake II Arena into CUDA C. This is only intended as a fun illustrative example;
# contemporary GPUs include direct instruction-level support for inverse square roots.
#
# The program computes the inverse square root of a collection of numbers
# and prints some of the results.
#
# We add the source code to our Modal environment by mounting the local files,
# adding them to the filesystem of the container running the function.
# We return the resulting binary -- it's just a stream of bytes.


@stub.function(
    gpu=GPU_CONFIG,
    image=cudatoolkit_image,
    mounts=[
        Mount.from_local_file("invsqrt_kernel.cu", "/root/invsqrt_kernel.cu"),
        Mount.from_local_file("invsqrt_demo.cu", "/root/invsqrt_demo.cu"),
    ],
)
def nvcc():  # TODO: use args to configure the nvcc command
    import subprocess
    from pathlib import Path

    assert not subprocess.run(
        ["nvcc", "-o", "invsqrt_demo", "invsqrt_kernel.cu", "invsqrt_demo.cu"],
        check=True,
    ).returncode

    assert not subprocess.run(
        ["./invsqrt_demo"], capture_output=True
    ).returncode

    return Path("./invsqrt_demo").read_bytes()


# The `main` function below puts this all together:
# it compiles the CUDA program, then shows the output of nvidia-smi
# in the environment it will be running in,
# then runs the program.
#
# You can pass the flag `--no-with-libcuda` to see what happens
# when the CUDA Driver API is removed.
# While nvidia-smi still runs, it no longer
# reports a CUDA version, and the program cannot run.


@stub.local_entrypoint()
def main(with_libcuda: bool = True):
    print(f"🔥 compiling our CUDA kernel on {GPU_CONFIG}")
    prog = nvcc.remote()
    with open("invsqrt_demo", "wb") as f:
        f.write(prog)

    if with_libcuda:
        print("🔥 showing nvidia-smi output")
        nvidia_smi.remote()
    else:
        print("🔥 removing libcuda to show what breaks")
        # remove_libcuda.remote()

    print("🔥 running our CUDA kernel")
    with open("invsqrt_demo", "rb") as f:
        prog = f.read()
    user_raw_cuda.remote(prog, with_libcuda)


@stub.function(gpu=GPU_CONFIG, image=cudatoolkit_image)
def user_raw_cuda(prog: bytes, with_libcuda: bool = True):
    import os
    import subprocess

    if not with_libcuda:
        remove_libcuda.local(verbose=False)

    with open("./prog", "wb") as f:
        f.write(prog)

    os.chmod("./prog", 0o755)
    subprocess.run(["./prog"])

    return
