set -e -x

ssh ubuntu@$CURRENT_INSTANCE 'echo connected'
rsync -a ./ ubuntu@$CURRENT_INSTANCE:/home/ubuntu/mojo
ssh ubuntu@$CURRENT_INSTANCE 'bash -c "cd /home/ubuntu/mojo && bash ./run-main.sh"'
rsync -a ubuntu@$CURRENT_INSTANCE:/home/ubuntu/mojo/July2024/benchmarks ./July2024/
