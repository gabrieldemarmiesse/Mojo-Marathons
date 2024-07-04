set -e -x

ssh ubuntu@$CURRENT_INSTANCE 'echo connected'
scp install-mojo.sh ubuntu@$CURRENT_INSTANCE:/home/ubuntu/install-mojo.sh