export OPENROUTER_API_KEY=$(cat shiran.api.secret)

if [ $# -eq 0 ]; then
    echo "No arguments provided. Please add model name via: ./collect-responses.sh <model-name>"
    exit 1
fi
echo "running collect-responses.py with --prompt-type: $1, --model: $2"
python collect-responses.py --prompt-type $1 --model $2
