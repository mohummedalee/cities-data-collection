export OPENROUTER_API_KEY=$(cat shiran.api.secret)

if [ $# -eq 0 ]; then
    echo "No arguments provided. Please add model name via: ./collect-responses.sh <model-name>"
    exit 1
fi
echo "calling collect-responses.py with model: $1"
python collect-responses.py --model $1
