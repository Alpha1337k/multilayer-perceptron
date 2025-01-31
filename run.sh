set -ex

docker build -t multilayer-perceptron .

docker run --rm -it -v $PWD:/app multilayer-perceptron $@