cd LLaMA-Factory

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

cd /xxx/LLaMA-Factory

uv tool install -e .

export TOOLBIN="$(dirname "$(readlink -f "$(command -v llamafactory-cli)")")"

export PATH="$TOOLBIN:$PATH"

"$TOOLBIN/python" -m ensurepip --upgrade

"$TOOLBIN/python" -m pip --version

"$TOOLBIN/python" -m pip install "deepspeed>=0.10.0,<0.17" bitsandbytes

"$TOOLBIN/python" -m pip install -U tensorboard tensorboardX

export PATH="/root/.local/share/uv/tools/llamafactory/bin:$PATH"

command -v torchrun && torchrun --help | head -n 3

"$TOOLBIN/python" -m pip install "transformers==4.51.1"