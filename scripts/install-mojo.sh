set -e -x
curl -s https://get.modular.com | sh -
modular auth examples
modular install mojo
BASHRC=$( [ -f "$HOME/.bash_profile" ] && echo "$HOME/.bash_profile" || echo "$HOME/.bashrc" )
echo 'export MODULAR_HOME="/home/ubuntu/.modular"' >> "$BASHRC"
echo 'export PATH="/home/ubuntu/.modular/pkg/packages.modular.com_mojo/bin:$PATH"' >> "$BASHRC"
source "$BASHRC"
