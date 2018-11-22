#!/usr/bin/env bash
# 2 layer experiments
# E
nsml run -d $1 -a 'dev --cuda --num_heads 2 --num_layers 2 --elmo --batch_size 32' --memory 16G

# E+D
nsml run -d $1 -a 'dev --cuda --num_heads 2 --num_layers 2 --elmo --dual --batch_size 32' --memory 16G

# E+S
nsml run -d $1 -a 'dev --cuda --num_heads 2 --num_layers 2 --elmo --sparse --batch_size 32' --memory 16G

# E+D+S
nsml run -d $1 -a 'dev --cuda --num_heads 2 --num_layers 2 --elmo --dual --sparse --batch_size 32' --memory 16G

# D
nsml run -d $1 -a 'dev --cuda --num_heads 2 --num_layers 2 --dual --preload' --memory 8G

# S
nsml run -d $1 -a 'dev --cuda --num_heads 2 --num_layers 2 --sparse --preload' --memory 8G

# S+D
nsml run -d $1 -a 'dev --cuda --num_heads 2 --num_layers 2 --dual --sparse --preload' --memory 8G
