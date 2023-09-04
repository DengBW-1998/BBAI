#!/usr/bin/env python

MODEL = ['adv', 'mlp']
RANDOM_SEED = 2
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.1
it = 3000
BATCH_SIZE = 32
EPOCHS = 9
HIDDEN_DIMENSIONS = 10  # hidden layer units for discriminator in GAN model
VALIDATE_ITER = 5  # validate the model every # iterations
DROPOUT = 0.5
GCN_OUTPUT_DIM = 10
ENCODER_HIDDEN_DIMENSIONS = 10
LATENT_DIMENSIONS = 8  # must be the same as the opposite set
DECODER_HIDDEN_DIMENSIONS = 10
MLP_HIDDEN_DIMENSIONS = 16

"""Parameters in writing and loading data to / from files."""
TRAINING_LOSS_PATH = 'metrics/experiments_results/decoder_training_loss.csv'
STEP1 = 'explicit_relation'
STEP2 = 'implicit_relation'
STEP3 = 'merge_relation'
STEP4 = 'opposite_relation'

# classification
method = "abcgraph"
input_folder = "./data/"
output_folder_bgnn_adv = "./out/bgnn-adv"
output_folder_bgnn_mlp = "./out/bgnn-mlp"
