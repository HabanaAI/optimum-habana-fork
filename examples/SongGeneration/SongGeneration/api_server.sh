export PYTHONPATH="$(pwd)/codeclm/tokenizer/":"$(pwd)":"$(pwd)/codeclm/tokenizer/Flow1dVAE/":"$(pwd)/codeclm/tokenizer/":$PYTHONPATH
PT_HPU_LAZY_MODE=1 python api_server.py --server-port 8485
