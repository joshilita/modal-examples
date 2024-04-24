# ---
# lambda-test: false
# ---
# # Stable Diffusion (A1111)
#
# This example runs the popular [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
# project on Modal, without modification. We just port the environment setup to a Modal container image
# and wrap the launch script with a `@web_server` decorator, and we're ready to go.
#
# You can run a temporary A1111 server with `modal serve a1111_webui.py` or deploy it permanently with `modal deploy a1111_webui.py`.

import subprocess

from modal import Image, Stub, web_server

PORT = 8000

# First, we define the image A1111 will run in.
# This takes a few steps because A1111 usually install its dependencies on launch via a script.
# The process may take a few minutes the first time, but subsequent image builds should only take a few seconds.

a12_image = (
    Image.debian_slim(python_version="3.11")
    .apt_install(
        "wget",
        "git",
        "libgl1",
        "libglib2.0-0",
        "google-perftools",  # For tcmalloc
    )
    .env({"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"})
    .run_commands(

        
        # "mkdir /yes && cd /yes && wget https://huggingface.co/mradermacher/Fimbulvetr-11B-v2-i1-GGUF/resolve/main/Fimbulvetr-11B-v2.i1-Q6_K.gguf?download=true -O model.gguf && curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64 && chmod +x koboldcpp",
        # "mkdir /yes && cd /yes && wget https://huggingface.co/Lewdiculous/Nyanade_Stunna-Maid-7B-v0.2-GGUF-IQ-Imatrix/resolve/main/Nyanade_Stunna-Maid-7B-v0.2-Q8_0-imat.gguf?download=true -O model.gguf && curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64 && chmod +x koboldcpp",
        "mkdir /yes && cd /yes && wget https://huggingface.co/LoneStriker/opus-v1.2-llama-3-8b-GGUF/resolve/main/opus-v1.2-llama-3-8b-Q6_K.gguf?download=true -O model.gguf && curl -fLo koboldcpp https://github.com/LostRuins/koboldcpp/releases/latest/download/koboldcpp-linux-x64 && chmod +x koboldcpp",
        "cd /yes && wget https://huggingface.co/cjpais/llava-1.6-mistral-7b-gguf/resolve/main/mmproj-model-f16.gguf?download=true -O other.gguf"
    )
)

stub = Stub("example-a2-webui", image=a12_image)

# After defining the custom container image, we start the server with `accelerate launch`. This
# function is also where you would configure hardware resources, CPU/memory, and timeouts.
#
# If you want to run it with an A100 or H100 GPU, just change `gpu="a10g"` to `gpu="a100"` or `gpu="h100"`.
#
# Startup of the web server should finish in under one to three minutes.


@stub.function(
    gpu="a10g",
    cpu=1,
    memory=20000,
    timeout=3600,
    # Allows 100 concurrent requests per container.
    allow_concurrent_inputs=100,
    container_idle_timeout=300,
    # Keep at least one instance of the server running.
    # keep_warm=1,
)
@web_server(port=PORT, startup_timeout=180)
def run():
    START_COMMAND = f"""
    cd /yes  && ./koboldcpp --model /yes/model.gguf --port 8000 --skiplauncher --host 0.0.0.0 --threads 2 --blasthreads 2 --nommap --usecublas --gpulayers 255 --highpriority --blasbatchsize 512 --contextsize 8192 --mmproj /yes/other.gguf


        
"""
    subprocess.Popen(START_COMMAND, shell=True)
