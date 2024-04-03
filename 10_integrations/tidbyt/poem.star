load("http.star", "http")
load("render.star", "render")

def fetch_poem():
    url = "https://modal-labs--tidbyt-app-poem.modal.run"
    response = http.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return response.body()

def main():
    poem = fetch_poem()
    lines = poem.splitlines()
    layout = render.Root(
        child=render.Column(
            children=
                [render.Marquee(child=render.Text(line, color="#7fee64", font="tom-thumb"), width=64) for line in lines]
                + [render.Text("- mixtral")]
        )
    )
    return layout

output = main()
