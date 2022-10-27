# stable-ouroboros

Infinite chains of captions and generations

## Usage

I recommend creating a new virtual environment and installing torch from
`pytorch.org` and requirements from `requirements.txt`. In particular, only
transformers version `4.17.0` is working right now.

Run `python ouroboros.py`. Supply either a `--prompt` or an `--image` path to
start the chain, e.g.

```
python ouroboros.py --prompt "Pretty painting of night sky" --n 20
python ouroboros.py --image examples/rain-princess.jpg --n 20
```

output (images, text file of prompts, and `index.html` visualizer) is saved to
`CWD/prompt_{prompt_name}` or `CWD/image_{image_basename}`.
