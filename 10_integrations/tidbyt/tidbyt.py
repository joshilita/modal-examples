from pathlib import Path

import modal

STARFILE = "poem.star"


image = (
    modal.Image.debian_slim(
        python_version="3.10"
    )  # start from a very minimal Linux image
    .apt_install("curl")  # add some system libraries or tools (here just curl)
    .run_commands(  # run shell setup commands
        [
            # download pixlet
            "curl -LO https://github.com/tidbyt/pixlet/releases/download/v0.22.4/pixlet_0.22.4_linux_amd64.tar.gz",
            # unpack it
            "tar -xvf pixlet_0.22.4_linux_amd64.tar.gz",
            # make it executable
            "chmod +x ./pixlet",
            # insert it into the path at the right spot
            "mv pixlet /usr/local/bin/pixlet",
            # and tidy up
            "rm pixlet_0.22.4_linux_amd64.tar.gz",
        ]
    )
    .pip_install(
        "requests", "pytz"
    )  # install Python packages, here the requests library for HTTP
)

stub = modal.Stub("tidbyt-app", image=image)


@stub.function(
    schedule=modal.Cron("*/5 * * * *"),
    mounts=[modal.Mount.from_local_file(STARFILE, "/root/starfile.star")],
    secrets=[modal.Secret.from_name("my-tidbyt-secret")],
)
def render():
    import os
    import subprocess

    DEVICE_ID = os.environ["TIDBYT_DEVICE_ID"]
    TIDBYT_API_KEY = os.environ["TIDBYT_API_KEY"]

    subprocess.run("pixlet render --gif starfile.star", check=True, shell=True)
    subprocess.run(
        f"pixlet push {DEVICE_ID} starfile.gif -t {TIDBYT_API_KEY}",
        check=True,
        shell=True,
    )


@stub.function(
    mounts=[modal.Mount.from_local_file(STARFILE, "/root/starfile.star")],
)
@modal.web_server(port=8080)
def preview():
    import subprocess

    subprocess.Popen("pixlet serve --host 0.0.0.0 starfile.star", shell=True)


poems = modal.Volume.from_name("poems", create_if_missing=True)
poems_path = Path("/root/poems")


@stub.function(volumes={poems_path: poems}, keep_warm=1)
@modal.web_endpoint()
def poem():
    year, month, day, hour = get_ymdh()

    poems.reload()  # refresh to see latest changes
    path = poems_path / year / month / day / hour / "poem.txt"
    print(f"retrieving poem from {path}")
    try:
        poem = path.read_text()
    except FileNotFoundError as e:
        poem = f"A message in red, so sudden and abrupt,\nFile not found, a poem corrupt?\nFIleNotFoundError: {e}"
    except Exception as e:
        poem = f"An unknown exception, caught by surprise,\nSilent code fails, under the user's eyes.\nException: {e}"

    return poem


@stub.function(schedule=modal.Cron("45 * * * *"), volumes={poems_path: poems})
def write_poem():
    import json
    from urllib.parse import quote

    # determine poem time
    year, month, day, hour = get_ymdh(next_hour=True)

    import requests

    poem_request = f"Write a poem in three lines, each with two or three words, that is fitting for the following date and time: Month #{month}, Day #{day}, Hour #{hour}."

    response = requests.get(
        f"https://modal-labs--vllm-mixtral.modal.run/completion/{quote(poem_request)}",
        stream=True,
    )

    poem = ""
    if response.status_code == 200:
        for chunk in response.iter_content(chunk_size=1024):
            chunk = chunk.decode("utf-8")
            for line in chunk.strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line[6:])
                        poem += data["text"]
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        print(line)

    poem = "\n".join(poem.strip().splitlines()[:3])
    print(poem)

    # write the poem to the volume
    path = poems_path / year / month / day / hour
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "poem.txt", "w") as f:
        f.write(poem)
    poems.commit()

    return poem


def get_ymdh(next_hour=False):
    import datetime

    import pytz

    eastern = pytz.timezone("America/New_York")

    # get year, month, day, and (possibly next) hour
    year, month, day, hour = (
        (
            datetime.datetime.now(eastern)
            + datetime.timedelta(hours=int(next_hour))
        )
        .strftime("%Y %m %d %H")
        .split()
    )

    return year, month, day, hour
