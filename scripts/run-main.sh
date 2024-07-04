set -e -x

export MODULAR_HOME="/home/ubuntu/.modular"
export PATH="/home/ubuntu/.modular/pkg/packages.modular.com_mojo/bin:$PATH"
cd /home/ubuntu/mojo/July2024
mojo build main.mojo
./main
