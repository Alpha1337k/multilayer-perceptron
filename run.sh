set -ex

docker build -t ft_dslr . 2> /dev/null

docker run --rm -it -v $PWD:/app ft_dslr $@