python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

if [ ! -f config/llm_config.json ]; then
    cp config/llm_config_template.json config/llm_config.json
    echo "Created config/llm_config.json from template. Please edit it with your OpenAI API key and model settings before running the tests."
    exit 1
fi

flake8
pytest --cov-branch --cov-report=term --cov=.
python main.py --test
