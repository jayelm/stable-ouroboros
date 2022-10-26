"""If ouroboros.py was changed, this grabs the list of prompts and reformats the HTML."""


from ouroboros import save_html_file
import os


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description='Reformatter',
        formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('dirname')

    args = parser.parse_args()

    with open(os.path.join(args.dirname, 'prompts.txt')) as f:
        prompts = f.read().strip().splitlines()

    save_html_file(prompts, args.dirname)
