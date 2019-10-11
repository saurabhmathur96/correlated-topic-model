import numpy as np
from scipy.optimize import minimize
np.random.seed(10)

def logistic_normal(mu, sigma):
  eta = np.random.multivariate_normal(mu, sigma)
  f_eta = f(eta)

  return f_eta

def sample_word(f_eta, beta):
  z = np.random.multinomial(1, pvals=f_eta).argmax()
  word = np.random.multinomial(1, pvals=beta[z]).argmax()
  return word

def random_model(topic_count, vocab_size):
  mu = np.random.uniform(25,1,topic_count-1)#np.ones(topic_count-1)
  mu /= sum(mu)
  # mu += 10
  #mu[1]= 10
  # mu[2]= 9
  #mu /= sum(mu)
  beta = np.random.uniform(0,1, (topic_count, vocab_size))
  
  for topic in range(topic_count):
    for n in range(vocab_size):
      beta[topic, n] += np.abs(np.random.normal(n, 1))

  for topic in range(topic_count):
    beta[topic] /= sum(beta[topic])
  #print (beta)
  sigma = np.eye(topic_count-1)*0.1

  return mu, sigma, beta


def model_param(topic_count, vocab_size):
  mu = np.zeros(topic_count-1)
  sigma = np.eye(topic_count-1)
  beta = 0.001 + np.random.uniform(0, 1, (topic_count, vocab_size))
  for i in range(topic_count):
    beta[i] /= sum(beta[i])

  return mu, np.linalg.inv(sigma), np.log(beta)

def var_param(topic_count, document_size):
  zeta = 10
  lam = np.zeros(topic_count)
  nu2 = np.ones(topic_count) 
  nu2[-1] = 0.0

  phi = np.ones((document_size, topic_count)) * (1/topic_count)
  return zeta, phi, lam, nu2


def entropy(doc, var):
  zeta, phi, lam, nu2 = var
  term1 = .5*sum(np.log(nu2[:-1] + np.log(2*np.pi) + 1))
  term2 = sum(np.dot(phi[n], np.log(phi[n])) for n, word in enumerate(doc))
  
  return term1 - term2

def lhood_bnd(doc, var, mod):
  det = np.linalg.det
  
  zeta, phi, lam, nu2 = var
  mu, sigma_inv, log_beta = mod
  
  topic_count = len(nu2)
  diff = lam[:-1] - mu
  nu2_diag = np.diag(nu2[:-1])
  
  term1 = 0.5*np.log(det(sigma_inv))
  term2 = 0.5*topic_count*np.log(2*np.pi)

  term3 = 0.5*(np.trace(np.dot(nu2_diag, sigma_inv)) + np.dot(np.dot(diff.T, sigma_inv), diff) )
  log_p_eta = term1 - term2 - term3
  
  term2 = (1/zeta)*sum(np.exp(lam[:-1] + nu2[:-1]/2)) 
  term3 = 1 - np.log(zeta)
  log_p_zn = len(doc)*(-term2 + term3)
  
  log_p_wn = 0
  for n, word in enumerate(doc):
    term1 = np.dot(lam[:-1], phi[n][:-1])   
    log_p_zn += (term1)
    
    log_p_wn += np.dot(phi[n], log_beta[:, word])
  
  return log_p_eta + log_p_wn + log_p_zn + entropy(doc, var)


def df_lam(doc, var, mod):
  zeta, phi, lam, nu2 = var
  mu, sigma_inv, log_beta = mod
  
  diff = lam.copy()
  diff[:-1] -= mu
  
  term1 = np.zeros((len(diff), len(diff)))
  term1[0:len(diff)-1, 0:len(diff)-1] = sigma_inv
  term1 = np.dot(term1, diff)
  
  term2 = sum(phi)
  #term2[-1] =0
  
  N = len(doc)
  term3 = (N/zeta)*(np.exp(lam + nu2/2))
  #term3[-1] = 0
  
  
  return -term1 + term2 - term3

def opt_lam(doc, var, mod):
  zeta, phi, lam, nu2 = var
  mu, sigma_inv, log_beta = mod
  
  fn = lambda x: -lhood_bnd(doc, (zeta, phi, x, nu2), mod)
  g = lambda x: -df_lam(doc, (zeta, phi, x, nu2), mod)
  
  #from scipy.optimize import approx_fprime
  #print (approx_fprime(lam, fn, 1e-9), g(lam))
  res = minimize(fn, x0=lam, jac=g, method='Newton-CG', options={'disp': 0})


  return res.x



def df_nu2_i(nu2i, i, doc, var, mod):
  zeta, phi, lam, _ = var
  mu, sigma_inv, log_beta = mod
  N = len(doc)
  
  term1 = 0.5*sigma_inv[i, i] 
  term2 = 0.5*(N/zeta)*np.exp(lam[i] + nu2i/2)
  term3 = 1/(2*nu2i)
  return -term1 -term2 + term3


def df2_nu2_i(nu2i, i, doc, var, mod):
  zeta, phi, lam, _ = var
  mu, sigma_inv, log_beta = mod
  N = len(doc)

  term1 = 0.25*(N/zeta) * np.exp(lam[i] + nu2i/2)
  term2 = 0.5*(1/(nu2i*nu2i))
  return -term1 -term2

def opt_nu2_i(g, h):
  init_x = 10
  x = init_x

  log_x = np.log(x)
  df1 = 1
  it = 0
  while np.abs(df1) > 0.0001:
    if np.isnan(x):
      init_x = init_x * 10
      x = init_x
      log_x = np.log(x)
    x = np.exp(log_x)

    df1 = g(x)
    df2 = h(x)

    log_x -= (x*df1)/(x*x*df2 + x*df1)
  return np.exp(log_x)  

def opt_nu2(doc, var, mod):
  zeta, phi, lam, nu2 = var
  mu, sigma_inv, log_beta = mod

  topic_count = len(nu2)
  for i in range(topic_count-1):
    g = lambda nu2i: df_nu2_i(nu2i, i, doc, (zeta, phi, lam, nu2), (mu, sigma_inv, log_beta))
    h = lambda nu2i: df2_nu2_i(nu2i, i, doc, (zeta, phi, lam, nu2), (mu, sigma_inv, log_beta))
    res = opt_nu2_i(g, h)
    nu2[i] = res
  return nu2

def opt_zeta(doc, var, mod):
  zeta, phi, lam, nu2 = var
  return sum(np.exp(lam + nu2/2)) + 1


def log_sum(log_a, log_b):
  if log_a < log_b:
    return log_b + np.log(1 + np.exp(log_a-log_b))
  else:
    return log_a + np.log(1 + np.exp(log_b-log_a))

def opt_phi(doc, var, mod):
  zeta, phi, lam, nu2 = var
  mu, sigma_inv, log_beta = mod
  
  log_phi = np.zeros_like(phi)
  nterms, ntopics = log_phi.shape
  for n, word in enumerate(doc):
    log_phi_sum = 0
    for i in range(ntopics):
      log_phi[n,i] = lam[i] + log_beta[i, word]
      if i == 0:
        log_sum_n = log_phi[n,i]
      else:
        log_sum_n = log_sum(log_sum_n, log_phi[n,i])
    
    for i in range(ntopics):
      log_phi[n,i] -= log_sum_n
      phi[n,i] = np.exp(log_phi[n,i])
  #print (log_phi)
  return phi


def inference(doc, mod):
  mu, sigma_inv, log_beta = mod

  topic_count = len(log_beta)
  document_size = len(doc)
  var = var_param(topic_count, document_size)
  zeta, phi, lam, nu2 = var

  
  for i in range(20):
    var = zeta, phi, lam, nu2
    lhood_old = lhood_bnd(doc, var, mod)
    # print ('lhood_old = ', lhood_old)

    
    var = zeta, phi, lam, nu2
    zeta = opt_zeta(doc, var, mod)

    var = zeta, phi, lam, nu2
    #print (lhood_bnd(doc, var, mod))
    
    var = zeta, phi, lam, nu2
    lam = opt_lam(doc, var, mod)
    lam[-1] = 0

    var = zeta, phi, lam, nu2
    #print (lhood_bnd(doc, var, mod))
    
    var = zeta, phi, lam, nu2
    zeta = opt_zeta(doc, var, mod)

    var = zeta, phi, lam, nu2
    #print (lhood_bnd(doc, var, mod))
    
    var = zeta, phi, lam, nu2
    nu2 = opt_nu2(doc, var, mod)
    
    var = zeta, phi, lam, nu2
    #print (lhood_bnd(doc, var, mod))

    var = zeta, phi, lam, nu2
    zeta = opt_zeta(doc, var, mod)

    var = zeta, phi, lam, nu2
    #print (lhood_bnd(doc, var, mod))


    var = zeta, phi, lam, nu2
    phi = opt_phi(doc, var, mod)

    var = zeta, phi, lam, nu2
    zeta = opt_zeta(doc, var, mod)

    var = zeta, phi, lam, nu2
    #print (lhood_bnd(doc, var, mod))

    var = zeta, phi, lam, nu2
    lhood = lhood_bnd(doc, var, mod)
    if ((lhood_old-lhood)/lhood_old) < 1e-6:
      break
    # print ('-lhood = ', lhood)
    

    lhood_old = lhood
  lam[-1] = 0
  return zeta, phi, lam, nu2


def f(y):
  out = np.zeros(len(y)+1)
  out[0:len(y)] = y
  return np.exp(out)/np.sum(np.exp(out))

def finv(x):
  x, xd = x[:-1], x[-1]
  return np.log(x/xd)

def main():
  topic_count = 4
  vocab_size = 10

  mod = random_model(topic_count, vocab_size)
  mu, sigma, beta = mod
  sigma_inv = np.linalg.inv(sigma)
  log_beta = np.log(beta)
  print (mu)
  print (f(mu))
  print (sigma)
  corpus_size = 100
  document_size = 50
  corpus = []
  fetas = []
  for d in range(corpus_size):
    eta = np.random.multivariate_normal(mu, sigma)
    f_eta = f(eta)
    fetas.append(f_eta.copy())
    print ('eta =', eta)
    print ('f(eta) =', f_eta)
    doc = []
    for i in range(document_size):
      z = np.random.multinomial(1, pvals=f_eta).argmax()
      # print ('z =', z)
      word = np.random.multinomial(1, pvals=beta[z]).argmax()
      doc.append(word)
      # print (word, end=' ')
    corpus.append(doc)
    # print (end='\n\n')

  lams = []
  phis = []
  nu2s = []
  mod = mu, sigma_inv, log_beta 
  for d in range(corpus_size):
    doc = corpus[d]
    var = inference(doc, mod)
    zeta, phi, lam, nu2 = var

    print ('document #%d'%d)
    print ('zeta = ', zeta)

    #print ('phi = ')
    #print (phi)
    
    print ('lam = ', lam)

    print ('nu2 = ', nu2)

    print ('f_lam = ', f(lam[:-1]))
    lams.append(lam.copy())
    phis.append(phi.copy())
    nu2s.append(nu2.copy())
  #print (mu)
  return np.array(lams), np.array(phis),np.array(fetas)


def expectation(corpus, mod):
  corpus_var = []
  for d, doc in enumerate(corpus):
    var = inference(doc, mod)
    zeta, phi, lam, nu2 = var
    corpus_var.append((zeta, phi.copy(), lam.copy(), nu2.copy()))
  return corpus_var

def maximization(corpus, corpus_var, vocab_size):
  lams= []
  nu2s = []
  phis = []
  for zeta, phi, lam, nu2 in corpus_var:
    lams.append(lam)
    nu2s.append(nu2)
    phis.append(phi)
  
  mu_sum = sum(lams)
  mu = mu_sum[:-1]/len(corpus)

  sigma_sum = sum(np.diag(nu2[:-1]) + np.outer(lam[:-1]-mu, lam[:-1]-mu) for lam, nu2 in zip(lams, nu2s))
  sigma = sigma_sum / len(corpus)
  sigma_inv = np.linalg.inv(sigma)
  # sigma_inv = np.eye(len(mu))

  topic_count = len(mu) + 1
  
  beta_ss = np.zeros((topic_count, vocab_size))
  for doc, phi in zip(corpus, phis):
    for i in range(topic_count):
      for n, word in enumerate(doc):
        beta_ss[i, word] += phi[n, i]


  log_beta = np.zeros((topic_count, vocab_size)) 
  for i in range(topic_count):
    sum_term = sum(beta_ss[i])

    if sum_term == 0:
      sum_term = (-1000)*vocab_size
      print (sum_term)
    else:
      sum_term = np.log(sum_term)

    for j in range(vocab_size):
      log_beta[i,j] = np.log(beta_ss[i,j]) - sum_term
  #print (np.exp(log_beta))
  return mu, sigma_inv, log_beta 



# main()

import os
from os import path
text = []
filenames = []
for file in sorted((fname for fname in os.listdir('20newsgroups') if not fname.endswith('.csv')), key=int):
  if file.endswith('csv'):
    continue
  filenames.append(file)
  file = open(path.join('20newsgroups', file), 'r')
  text.append(file.read().split())

words = sorted(set(sum(text, [])))
documents = []
for row in text:
  documents.append([words.index(word) for word in row])


open('words.txt', 'w').write(' '.join(words))

topic_count = 20
mod = model_param(topic_count, len(words))
# mu, sigma_inv, log_beta = mod

after = sum(lhood_bnd(doc, var_param(topic_count, len(doc)), mod) for doc in documents)
print ('init ', after)
for _ in range(1000):
  before = after
  corpus_var = expectation(documents, mod)
  mod = maximization(documents, corpus_var, len(words))
  
  
  after = sum(lhood_bnd(doc, var, mod) for doc, var in zip(documents,corpus_var) )
  print ('lhood = ', after)
  print (((before - after) / before))
  if ((before - after) / before) < 0.001:
    break
mu, sigma_inv, log_beta = mod
np.savetxt('beta.txt', np.exp(log_beta))
corpus_lam = np.array([lam for zeta, phi, lam, nu2  in corpus_var])
np.savetxt('corpus-lam.txt', corpus_lam)
# main2()