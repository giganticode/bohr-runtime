#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

base_ref="$1"

if [ -z "$base_ref" ]; then
    base_ref="origin/master"
    echo "base_ref not passed - comparing Dockerfile with default base_ref: $base_ref"
fi

cmd="git --no-pager diff $base_ref -- Dockerfile"
echo "Running command: $cmd"

DOCKERFILE_DIFF="$(git --no-pager diff $base_ref -- Dockerfile)"

if [ -n "$DOCKERFILE_DIFF" ]; then
    echo "Dockerfile has changed"
    echo "The change is the following:"
    echo "----------------------------------------------------------"
    echo "$DOCKERFILE_DIFF"
    echo "----------------------------------------------------------"
    docker build --tag giganticode/bohr-cml-base:latest .
    docker push giganticode/bohr-cml-base:latest
else
    echo "Dockerfile hasn't change."
fi
