### Setting up
To set up and activate the conda environment for running scripts, use:

```
conda create -f env.yaml
conda activate cities-auditing
```

### Running data collection
Make sure you have Shiran's OpenRouter API key in a file called `shiran.api.secret`, it is not available on this repo.
To run data collection, simply run: `./collect-responses.sh <prompt_type> <model_type>`

For example, to collect responses for all `generic` prompts with LLaMa, you would do: `./collect-responses.sh generic llama`. Responses will be saved in `responses/generic/llama.csv`.