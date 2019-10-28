import numpy as np
import pymc3 as pm

def model(documents, word_count, topic_count):
  ''' initializes the model parameters: theta (doc-topic), phi(topic-word) '''
  pass

def mcmc_inference(documents, model, alpha, beta, iteration_count):
  ''' performs inference on lda model by gibbs sampling '''
  with pm.Model() as lda_model:
    # model definition
    # pm.sample(1000)
    pass
