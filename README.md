# Connect4 Bot - CNN and Transformer

This project trains two neural networks to play Connect4:

- A deep CNN with residual + SE blocks
- A Vision Transformer style model

Both models take a 6x7 board encoded as 2 channels and predict the best column to play.

## Structure
- app/ : model architectures
- weights/ : trained checkpoints
- inference.py : prediction API
- test_inference.py : quick tests

## Usage
Run test_inference.py to see predictions.
