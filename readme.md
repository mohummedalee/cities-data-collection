### Setting up
To set up and activate the conda environment for running scripts, use:

```
conda create -f env.yaml
conda activate cities-auditing
```

### Running data collection
Ensure you have your own OpenRouter API key in a file called `api.secret`, it is not available on this repo.
To run data collection, simply run: `./collect-responses.sh <prompt_type> <model_type> <n_samples>`

For example, to collect responses for all `generic` prompts with LLaMa, you would do: `./collect-responses.sh generic llama 20`. Responses will be saved in `responses/generic/llama.csv`.
