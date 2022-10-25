"""Stable Ouroboros

Start with an image, get a prompt for the image, generate a new image based on
the prompt, rinse and repeat.
"""


import os

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from tqdm import tqdm, trange


HTML_SKELETON = """
<!DOCTYPE html>
<html>
  <head>
    <title>{directory}</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap.min.css" integrity="sha384-HSMxcRTRxnN+Bdg0JdbxYKrThecOKuH5zCYotlSAcp1+c8xmyTe9GYg1l9a69psu" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@3.4.1/dist/css/bootstrap-theme.min.css" integrity="sha384-6pzBo3FDv/PJ8r2KRkGHifhEocL+1X2rVCTTkUfGk7/0pbek5mMa1upzvWbrUbOZ" crossorigin="anonymous">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@splidejs/splide@4.1.3/dist/css/splide.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Roboto+Mono&display=swap" rel="stylesheet"><Paste>
    <style>
    p {{
	font-family: 'Roboto Mono', monospace;
        font-size: 18px;
    }}
    .splide__slide img {{
        width: 100%;
        height: auto;
    }}
    </style>
  </head>
  <body>
    <section class="splide" aria-label="Splide Basic HTML Example">
  <div class="splide__track">
		<ul class="splide__list">
                {splide_htmls}
		</ul>
  </div>
</section>
    <script src="https://cdn.jsdelivr.net/npm/@splidejs/splide@4.1.3/dist/js/splide.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@splidejs/splide-extension-auto-scroll@0.5.3/dist/js/splide-extension-auto-scroll.min.js"></script>
    <script>
  document.addEventListener( 'DOMContentLoaded', function() {{
    var splide = new Splide( '.splide' , {{
    autoScroll: {{speed: 2 }},
    perPage: 3,
    drag   : 'free',
    focus  : 'center',
    }});
    splide.mount( window.splide.Extensions );
  }} );
</script>
  </body>
</html>
"""

SPLIDE_SKELETON = "<li class=\"splide__slide\">{html}</li>"


HTML_TEMPLATE = """
<dir class="iteration">
    <img class="image" src="{image}"></img>
    <p class="caption">{iteration}. {prompt}</p>
</dir>
"""


def is_nsfw_image(pil_image):
    x = np.array(pil_image)
    if (x == 0).all():
        return True
    return False


def save_prompt_and_image(i, prompt, image, directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "prompts.txt"), "a") as f:
        f.write(prompt + "\n")
    image.save(os.path.join(directory, f"{i}.png"))


def save_html_file(prompts, directory):
    os.makedirs(directory, exist_ok=True)
    htmls = []
    for i, prompt in enumerate(prompts):
        html_snippet = HTML_TEMPLATE.format(
            image=f"{i}.png",
            prompt=prompt,
            iteration=i,
        )
        html_snippet = SPLIDE_SKELETON.format(
            html=html_snippet
        )
        htmls.append(html_snippet)
    html = HTML_SKELETON.format(
        directory=directory,
        splide_htmls="\n".join(htmls)
    )

    with open(os.path.join(directory, "index.html"), "w") as f:
        f.write(html + "\n")


def sanitize_filename(s):
    return "".join([c if c.isalnum() else "_" for c in s]).lower()


def main(args):
    # Load interrogate only when needed.
    from interrogator import interrogate

    sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_type=torch.float16, revision="fp16"
    )
    sd = sd.to("cuda")

    if args.prompt is not None:
        # Generate first image, discard first prompt.
        image = sd(args.prompt).images[0]
        dirname = f"prompt_{sanitize_filename(args.prompt)[:50]}"
    else:
        assert args.image is not None
        image = Image.open(args.image)
        dirname = f"image_{os.path.basename(image)}"

    prompts = []
    images = []
    for i in trange(args.n, desc="Ouroboros"):
        prompt = interrogate(image)
        prompts.append(prompt)
        images.append(image)
        tqdm.write(prompt)
        #  Save this (image, prompt) pair.
        save_prompt_and_image(i, prompt, image, dirname)
        save_html_file(prompts, dirname)
        image = sd(prompt).images[0]
        n_tries = 0
        while is_nsfw_image(image):
            if n_tries >= args.max_retries:
                raise RuntimeError(
                    f"Tried {args.max_retries} attempts to generate non-NSFW"
                    "image, but failed"
                )
            image = sd(prompt).images[0]
            print("Got NSFW image, retrying")
            n_tries += 1


if __name__ == "__main__":
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description="Stable Ouroboros", formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--image", type=str, default=None, help="Initial image")
    parser.add_argument("--prompt", type=str, default=None, help="Initial prompt")
    parser.add_argument("--n", type=int, default=10, help="Number of loops to run")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of times to retry generation if NSFW image generated",
    )

    args = parser.parse_args()

    if (args.image is None and args.prompt is None) or (
        args.image is not None and args.prompt is not None
    ):
        parser.error("must supply exactly one of --image or --prompt.")

    main(args)
