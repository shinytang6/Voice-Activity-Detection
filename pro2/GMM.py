from sklearn.mixture import GaussianMixture, gaussian_mixture
import numpy as np
from htkread import *

x = readhtk("chen_0004092_B.mfcc")
# sil
sil_mean = np.loadtxt('sil_mean.txt')
sil_variance = np.loadtxt('sil_variance.txt')
sil_weight = np.loadtxt('sil_weight.txt')
sil_gmm = GaussianMixture(128,covariance_type="diag")
sil_precisions_cholesky_ = gaussian_mixture._compute_precision_cholesky(sil_variance,"diag")

sil_gmm.means_ = sil_mean
sil_gmm.weights_ = sil_weight
sil_gmm.precisions_cholesky_ = sil_precisions_cholesky_
sil_result = np.dot(sil_gmm.predict_proba(x),sil_weight.reshape(-1,1))
print(sil_result)


# speech
speech_mean = np.loadtxt('speech_mean.txt')
speech_variance = np.loadtxt('speech_variance.txt')
speech_weight = np.loadtxt('speech_weight.txt')
speech_gmm = GaussianMixture(128,covariance_type="diag")
speech_precisions_cholesky_ = gaussian_mixture._compute_precision_cholesky(speech_variance,"diag")

speech_gmm.means_ = speech_mean
speech_gmm.weights_ = speech_weight
speech_gmm.precisions_cholesky_ = speech_precisions_cholesky_
speech_result = np.dot(speech_gmm.predict_proba(x),speech_weight.reshape(-1,1))
print(speech_result)

# noise
noise_mean = np.loadtxt('noise_mean.txt')
noise_variance = np.loadtxt('noise_variance.txt')
noise_weight = np.loadtxt('noise_weight.txt')
noise_gmm = GaussianMixture(128,covariance_type="diag")
noise_precisions_cholesky_ = gaussian_mixture._compute_precision_cholesky(noise_variance,"diag")

noise_gmm.means_ = noise_mean
noise_gmm.weights_ = noise_weight
noise_gmm.precisions_cholesky_ = noise_precisions_cholesky_
noise_result = np.dot(noise_gmm.predict_proba(x),noise_weight.reshape(-1,1))
print(noise_result)

print(noise_result.shape)
output = open("output.txt",'w')
start = 0
for seg in range(179998):
    sil = sil_result[seg]
    speech = speech_result[seg]
    noise = noise_result[seg]
    end = start + 25
    if(sil>speech):
        if(sil>noise):
            output.write(str(start)+" "+ str(end)+ " "+ "sil" + "\n")
        else:
            output.write(str(start)+" "+ str(end)+ " "+ "noise" + "\n")
    else:
        if(speech>noise):
            output.write(str(start)+" "+ str(end)+ " "+ "speech" + "\n")
        else: 
            output.write(str(start)+" "+ str(end)+ " "+ "noise" + "\n")
    start = start + 10
