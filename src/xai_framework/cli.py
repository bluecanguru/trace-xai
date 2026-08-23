import argparse
import yaml
from core.experiment import XAIExperiment


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    config = load_config(args.config)

    exp = XAIExperiment(config)

    exp.run_global()
    exp.run_local()
    exp.run_stability()

    exp.export(config.get("output_dir", "results/out"))


if __name__ == "__main__":
    main()