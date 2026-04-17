python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

if [ ! -f config/llm_config.json ]; then
    cp config/llm_config_template.json config/llm_config.json
    echo "Created config/llm_config.json from template. Please edit it with your OpenAI API key and model settings before running the tests."
    exit 1
fi

# 只扫描项目代码，排除虚拟环境和第三方包
flake8 --exclude=.venv,tests,docs neuro_agent_framework examples
pytest --cov-branch --cov-report=term --cov=.
python framework_test.py -n 1
