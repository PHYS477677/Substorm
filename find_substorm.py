# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 13:14:30 2021

@author: Yiwen Zhu
"""
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# be sure to load eli5 module
from eli5.permutation_importance import get_score_importances
from sklearn.metrics import accuracy_score
# %% read in data

# read in SMLde data

path_SML = r'C:\Users\Chisei\OneDrive - Rice University\work\study\data science in physics\data\supermag.dat'
f_SML = open(path_SML, 'r')
next(f_SML)
data_SML = [re.split(r'\t+', iline) for iline in f_SML.readlines()]
data_SML = np.array(data_SML, dtype=float)
SML = data_SML[:, -2]
year = data_SML[:, 0]
month = data_SML[:, 1]
day = data_SML[:, 2]
hour = data_SML[:, 3]
minute = data_SML[:, 4]
time = list((zip(year, month, day, hour, minute)))

# read in OMNI data

path_OMNI = r'C:\Users\Chisei\OneDrive - Rice University\work\study\data science in physics\data\OMNI.dat'
f_OMNI = open(path_OMNI, 'r')
data_OMNI = [iline.split() for iline in f_OMNI.readlines()]
data_OMNI = np.array(data_OMNI, dtype=float)
Bx = data_OMNI[:, 4]
By = data_OMNI[:, 5]
Bz = data_OMNI[:, 6]
vel = data_OMNI[:, 7]
rho_n = data_OMNI[:, 8]
pres = data_OMNI[:, 10]
# ! replace the unreasonably high value, these value means no observation there
Bx[np.where(abs(Bx) >= 10)] = np.nan
By[np.where(abs(By) >= 10)] = np.nan
Bz[np.where(abs(Bz) >= 10)] = np.nan
vel[np.where(abs(vel) >= 10000)] = np.nan
rho_n[np.where(abs(rho_n) >= 100)] = np.nan
pres[np.where(abs(pres > 10))] = np.nan

# read in substorm lists

path_list = r'C:\Users\Chisei\OneDrive - Rice University\work\study\data science in physics\data\substorms-list.ascii'
f_list = open(path_list, 'r', encoding='UTF-8')
for i in range(38):
    next(f_list)
data_list = [iline.split() for iline in f_list.readlines()]
data_list = np.array(data_list, dtype=float)
# ! since the first substorm's onset's time difference from the data starting point is less than 120 minutes, we delete it.
data_list = np.delete(data_list, 0, axis=0)
year_substorm = data_list[:, 0]
month_substorm = data_list[:, 1]
day_substorm = data_list[:, 2]
hour_substorm = data_list[:, 3]
minute_substorm = data_list[:, 4]
time_substorm = list((zip(year_substorm, month_substorm,
                          day_substorm, hour_substorm, minute_substorm)))


# %% find substorm and nosubstorm

# find substorm

# find the var in the list
# SML for substorm, full means the time period is 60 min before the onset, 120 minutes after the onset.
SML_substorm_full = []
Bx_substorm_full = []
By_substorm_full = []
Bz_substorm_full = []
vel_substorm_full = []
rho_n_substorm_full = []
pres_substorm_full = []
idx_substorm_full_flat = []  # indices that correspondes to the substorms
ctr = 1

for i in time_substorm:
    if ctr % int(0.1*len(time_substorm)) == 0:
        print('finished percentile of finding substorm: ',
              (ctr/len(time_substorm)))
    idx = time.index(i)  # index for each substorm onset
    SML_substorm_full.append(SML[idx-60:idx+120])
    Bx_substorm_full.append(Bx[idx-60:idx+120])
    By_substorm_full.append(By[idx-60:idx+120])
    Bz_substorm_full.append(Bz[idx-60:idx+120])
    vel_substorm_full.append(vel[idx-60:idx+120])
    rho_n_substorm_full.append(rho_n[idx-60:idx+120])
    pres_substorm_full.append(pres[idx-60:idx+120])
    idx_substorm_full_flat += range(idx-120, idx+120)
    ctr += 1
# save data
np.save('../output/Bx_substorm_full', Bx_substorm_full)
np.save('../output/By_substorm_full', By_substorm_full)
np.save('../output/Bz_substorm_full', Bz_substorm_full)
np.save('../output/vel_substorm_full', vel_substorm_full)
np.save('../output/rho_n_substorm_full', rho_n_substorm_full)
np.save('../output/pres_substorm_full', pres_substorm_full)
np.save('../output/SML_substorm_full', SML_substorm_full)
np.save('../output/idx_substorm_full', idx_substorm_full_flat)

# epoch analysis
SML_substorm_mean = np.nanmean(SML_substorm_full, 0)
Bx_substorm_mean = np.nanmean(Bx_substorm_full, 0)
By_substorm_mean = np.nanmean(By_substorm_full, 0)
Bz_substorm_mean = np.nanmean(Bz_substorm_full, 0)
vel_substorm_mean = np.nanmean(vel_substorm_full, 0)
rho_n_substorm_mean = np.nanmean(rho_n_substorm_full, 0)
pres_substorm_mean = np.nanmean(pres_substorm_full, 0)


length = len(Bx_substorm_full[0])
plt.close('all')
figs, axes = plt.subplots(7, 1, figsize=(20, 15))
plt.setp(axes, xticks=np.linspace(0, length, 4), xticklabels=[
    '-01:00', '00:00', '01:00', '02:00'], xlabel='Epoch Time(h)')
plt.xlabel('Universal Time(h)', fontsize=20, labelpad=20)
plt.sca(axes[0])
plt.plot(range(length), Bx_substorm_mean)
plt.ylabel('IMF Bx/nT', rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[1])
plt.plot(range(length), By_substorm_mean)
plt.ylabel('IMF By/nT', rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[2])
plt.plot(range(length), Bz_substorm_mean)
plt.ylabel('IMF Bz/nT', rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[3])
plt.plot(range(length), vel_substorm_mean)
plt.ylabel('Solar Wind \n Velocity \n (km/s)',
           rotation=0, labelpad=80, fontsize=15)

plt.sca(axes[4])
plt.plot(range(length), rho_n_substorm_mean)
plt.ylabel('Solar Wind \n Number Density\n (n/cc)',
           rotation=0, labelpad=80, fontsize=15)

plt.sca(axes[5])
plt.plot(range(length), pres_substorm_mean)
plt.ylabel('Solar Wind Pressure\n (nPa)', rotation=0, labelpad=85, fontsize=15)

plt.sca(axes[6])
plt.plot(range(length), SML_substorm_mean)
plt.ylabel('SML\n (nT)', rotation=0, labelpad=30, fontsize=15)

plt.suptitle('Epoch Analysis', fontsize=20)

# find the non substorm

# get the variables till the onset
Bx_substorm = [i[0:int(length/3)] for i in Bx_substorm_full]
By_substorm = [i[0:int(length/3)] for i in By_substorm_full]
Bz_substorm = [i[0:int(length/3)] for i in Bz_substorm_full]
vel_substorm = [i[0:int(length/3)] for i in vel_substorm_full]
rho_n_substorm = [i[0:int(length/3)] for i in rho_n_substorm_full]
pres_substorm = [i[0:int(length/3)] for i in pres_substorm_full]

per_substorm = 75/100
per_nosubstorm = 1-per_substorm
num_substorm = len(Bx_substorm)
num_nosubstorm = int(num_substorm*per_nosubstorm/per_substorm)
num_tot = num_substorm+num_nosubstorm

# prepare nonstorm case
idx_whole = range(len(SML))
SML_nosubstorm = []
Bx_nosubstorm = []
By_nosubstorm = []
Bz_nosubstorm = []
vel_nosubstorm = []
rho_n_nosubstorm = []
pres_nosubstorm = []

A = np.array(idx_whole)
B = np.array(idx_substorm_full_flat)
# 　way to remove sorted elements in one array from another sorted array
idx_list_nosubstorm = A[~np.in1d(A, B)]
# delete the last non substorm case to avoid out of the boundary
idx_list_nosubstorm = np.delete(idx_list_nosubstorm, range(-60, 0), axis=0)
for i in range(num_nosubstorm):
    idx_start = random.choice(idx_list_nosubstorm)
    idx_range_nosubstorm = range(idx_start, idx_start+60)
    SML_nosubstorm.append(SML[idx_range_nosubstorm])
    Bx_nosubstorm.append(Bx[idx_range_nosubstorm])
    By_nosubstorm.append(By[idx_range_nosubstorm])
    Bz_nosubstorm.append(Bz[idx_range_nosubstorm])
    vel_nosubstorm.append(vel[idx_range_nosubstorm])
    rho_n_nosubstorm.append(rho_n[idx_range_nosubstorm])
    pres_nosubstorm.append(pres[idx_range_nosubstorm])

# %% process and get the train data

# address the nan
# converet list to array
Bx_substorm = np.array(Bx_substorm)
Bx_nosubstorm = np.array(Bx_nosubstorm)
By_substorm = np.array(By_substorm)
By_nosubstorm = np.array(By_nosubstorm)
Bz_substorm = np.array(Bz_substorm)
Bz_nosubstorm = np.array(Bz_nosubstorm)
vel_substorm = np.array(vel_substorm)
vel_nosubstorm = np.array(vel_nosubstorm)
rho_n_substorm = np.array(rho_n_substorm)
rho_n_nosubstorm = np.array(rho_n_nosubstorm)
pres_substorm = np.array(pres_substorm)
pres_nosubstorm = np.array(pres_nosubstorm)

# combine substorm and non substorm cases
Bx_cases = np.vstack((Bx_substorm, Bx_nosubstorm))  # Bz for all cases
By_cases = np.vstack((By_substorm, By_nosubstorm))  # Bz for all cases
Bz_cases = np.vstack((Bz_substorm, Bz_nosubstorm))  # Bz for all cases
vel_cases = np.vstack((vel_substorm, vel_nosubstorm))  # Bz for all cases
rho_n_cases = np.vstack((rho_n_substorm, rho_n_nosubstorm))  # Bz for all cases
pres_cases = np.vstack((pres_substorm, pres_nosubstorm))  # Bz for all cases

# interplote
Bx_cases = pd.DataFrame(Bx_cases)
Bx_cases = pd.DataFrame.interpolate(Bx_cases, axis=1)
Bx_cases = np.array(Bx_cases)
Bx_substorm = Bx_cases[:num_substorm]
Bx_nosubstorm = Bx_cases[num_substorm:]

By_cases = pd.DataFrame(By_cases)
By_cases = pd.DataFrame.interpolate(By_cases, axis=1)
By_cases = np.array(By_cases)
By_substorm = By_cases[:num_substorm]
By_nosubstorm = By_cases[num_substorm:]

Bz_cases = pd.DataFrame(Bz_cases)
Bz_cases = pd.DataFrame.interpolate(Bz_cases, axis=1)
Bz_cases = np.array(Bz_cases)
Bz_substorm = Bz_cases[:num_substorm]
Bz_nosubstorm = Bz_cases[num_substorm:]

vel_cases = pd.DataFrame(vel_cases)
vel_cases = pd.DataFrame.interpolate(vel_cases, axis=1)
vel_cases = np.array(vel_cases)
vel_substorm = vel_cases[:num_substorm]
vel_nosubstorm = vel_cases[num_substorm:]

rho_n_cases = pd.DataFrame(rho_n_cases)
rho_n_cases = pd.DataFrame.interpolate(rho_n_cases, axis=1)
rho_n_cases = np.array(rho_n_cases)
rho_n_substorm = rho_n_cases[:num_substorm]
rho_n_nosubstorm = rho_n_cases[num_substorm:]

pres_cases = pd.DataFrame(pres_cases)
pres_cases = pd.DataFrame.interpolate(pres_cases, axis=1)
pres_cases = np.array(pres_cases)
pres_substorm = pres_cases[:num_substorm]
pres_nosubstorm = pres_cases[num_substorm:]

# after interpolating, assign a typical row to the rows containing nan
row_substorm_nonan = ~(np.isnan(Bx_substorm).any(axis=1))
row_substorm_nonan = [i for i, x in enumerate(row_substorm_nonan) if x]
# Bx, a random storm case with no nan
Bx_substorm_nonan = Bx_substorm[random.choice(row_substorm_nonan)]
row_substorm_nan = np.isnan(Bx_substorm).any(axis=1)
row_substorm_nan = [i for i, x in enumerate(row_substorm_nan) if x]
Bx_substorm[row_substorm_nan] = Bx_substorm_nonan

row_nosubstorm_nonan = ~(np.isnan(Bx_nosubstorm).any(axis=1))
row_nosubstorm_nonan = [i for i, x in enumerate(row_nosubstorm_nonan) if x]
Bx_nosubstorm_nonan = Bx_nosubstorm[random.choice(row_nosubstorm_nonan)]
row_nosubstorm_nan = np.isnan(Bx_nosubstorm).any(axis=1)
row_nosubstorm_nan = [i for i, x in enumerate(row_nosubstorm_nan) if x]
Bx_nosubstorm[row_nosubstorm_nan] = Bx_nosubstorm_nonan

row_substorm_nonan = ~(np.isnan(By_substorm).any(axis=1))
row_substorm_nonan = [i for i, x in enumerate(row_substorm_nonan) if x]
By_substorm_nonan = By_substorm[random.choice(row_substorm_nonan)]
row_substorm_nan = np.isnan(By_substorm).any(axis=1)
row_substorm_nan = [i for i, x in enumerate(row_substorm_nan) if x]
By_substorm[row_substorm_nan] = By_substorm_nonan

row_nosubstorm_nonan = ~(np.isnan(By_nosubstorm).any(axis=1))
row_nosubstorm_nonan = [i for i, x in enumerate(row_nosubstorm_nonan) if x]
By_nosubstorm_nonan = By_nosubstorm[random.choice(row_nosubstorm_nonan)]
row_nosubstorm_nan = np.isnan(By_nosubstorm).any(axis=1)
row_nosubstorm_nan = [i for i, x in enumerate(row_nosubstorm_nan) if x]
By_nosubstorm[row_nosubstorm_nan] = By_nosubstorm_nonan

row_substorm_nonan = ~(np.isnan(Bz_substorm).any(axis=1))
row_substorm_nonan = [i for i, x in enumerate(row_substorm_nonan) if x]
Bz_substorm_nonan = Bz_substorm[random.choice(row_substorm_nonan)]
row_substorm_nan = np.isnan(Bz_substorm).any(axis=1)
row_substorm_nan = [i for i, x in enumerate(row_substorm_nan) if x]
Bz_substorm[row_substorm_nan] = Bz_substorm_nonan

row_nosubstorm_nonan = ~(np.isnan(Bz_nosubstorm).any(axis=1))
row_nosubstorm_nonan = [i for i, x in enumerate(row_nosubstorm_nonan) if x]
Bz_nosubstorm_nonan = Bz_nosubstorm[random.choice(row_nosubstorm_nonan)]
row_nosubstorm_nan = np.isnan(Bz_nosubstorm).any(axis=1)
row_nosubstorm_nan = [i for i, x in enumerate(row_nosubstorm_nan) if x]
Bz_nosubstorm[row_nosubstorm_nan] = Bz_nosubstorm_nonan

vel_cases = pd.DataFrame(vel_cases, dtype=float)
vel_cases = pd.DataFrame.interpolate(vel_cases, axis=1)
vel_cases = np.array(vel_cases)

row_substorm_nonan = ~(np.isnan(vel_substorm).any(axis=1))
row_substorm_nonan = [i for i, x in enumerate(row_substorm_nonan) if x]
vel_substorm_nonan = vel_substorm[random.choice(row_substorm_nonan)]
row_substorm_nan = np.isnan(vel_substorm).any(axis=1)
row_substorm_nan = [i for i, x in enumerate(row_substorm_nan) if x]
vel_substorm[row_substorm_nan] = vel_substorm_nonan

row_nosubstorm_nonan = ~(np.isnan(vel_nosubstorm).any(axis=1))
row_nosubstorm_nonan = [i for i, x in enumerate(row_nosubstorm_nonan) if x]
vel_nosubstorm_nonan = vel_nosubstorm[random.choice(
    row_nosubstorm_nonan)]
row_nosubstorm_nan = np.isnan(vel_nosubstorm).any(axis=1)
row_nosubstorm_nan = [i for i, x in enumerate(row_nosubstorm_nan) if x]
vel_nosubstorm[row_nosubstorm_nan] = vel_nosubstorm_nonan

rho_n_cases = pd.DataFrame(rho_n_cases, dtype=float)
rho_n_cases = pd.DataFrame.interpolate(rho_n_cases, axis=1)
rho_n_cases = np.array(rho_n_cases)

row_substorm_nonan = ~(np.isnan(rho_n_substorm).any(axis=1))
row_substorm_nonan = [i for i, x in enumerate(row_substorm_nonan) if x]
rho_n_substorm_nonan = rho_n_substorm[random.choice(row_substorm_nonan)]
row_substorm_nan = np.isnan(rho_n_substorm).any(axis=1)
row_substorm_nan = [i for i, x in enumerate(row_substorm_nan) if x]
rho_n_substorm[row_substorm_nan] = rho_n_substorm_nonan

row_nosubstorm_nonan = ~(np.isnan(rho_n_nosubstorm).any(axis=1))
row_nosubstorm_nonan = [i for i, x in enumerate(row_nosubstorm_nonan) if x]
rho_n_nosubstorm_nonan = rho_n_nosubstorm[random.choice(
    row_nosubstorm_nonan)]  # rho_n, a random storm case with no nan
row_nosubstorm_nan = np.isnan(rho_n_nosubstorm).any(axis=1)
row_nosubstorm_nan = [i for i, x in enumerate(row_nosubstorm_nan) if x]
rho_n_nosubstorm[row_nosubstorm_nan] = rho_n_nosubstorm_nonan

pres_cases = pd.DataFrame(pres_cases, dtype=float)
pres_cases = pd.DataFrame.interpolate(pres_cases, axis=1)
pres_cases = np.array(pres_cases)

row_substorm_nonan = ~(np.isnan(pres_substorm).any(axis=1))
row_substorm_nonan = [i for i, x in enumerate(row_substorm_nonan) if x]
pres_substorm_nonan = pres_substorm[random.choice(row_substorm_nonan)]
row_substorm_nan = np.isnan(pres_substorm).any(axis=1)
row_substorm_nan = [i for i, x in enumerate(row_substorm_nan) if x]
pres_substorm[row_substorm_nan] = pres_substorm_nonan

row_nosubstorm_nonan = ~(np.isnan(pres_nosubstorm).any(axis=1))
row_nosubstorm_nonan = [i for i, x in enumerate(row_nosubstorm_nonan) if x]
pres_nosubstorm_nonan = pres_nosubstorm[random.choice(
    row_nosubstorm_nonan)]  # pres, a random storm case with no nan
row_nosubstorm_nan = np.isnan(pres_nosubstorm).any(axis=1)
row_nosubstorm_nan = [i for i, x in enumerate(row_nosubstorm_nan) if x]
pres_nosubstorm[row_nosubstorm_nan] = pres_nosubstorm_nonan

# combine the storm and nostorm cases again
Bx_cases = np.vstack((Bx_substorm, Bx_nosubstorm))  # Bx for all cases
By_cases = np.vstack((By_substorm, By_nosubstorm))  # By for all cases
Bz_cases = np.vstack((Bz_substorm, Bz_nosubstorm))  # Bz for all cases
vel_cases = np.vstack((vel_substorm, vel_nosubstorm))  # vel for all cases
rho_n_cases = np.vstack((rho_n_substorm, rho_n_nosubstorm))
pres_cases = np.vstack((pres_substorm, pres_nosubstorm))  # pres for all cases

# save the data
np.save('../output/Bx_substorm', Bx_substorm)
np.save('../output/By_substorm', By_substorm)
np.save('../output/Bz_substorm', Bz_substorm)
np.save('../output/vel_substorm', vel_substorm)
np.save('../output/rho_n_substorm', rho_n_substorm)
np.save('../output/pres_substorm', pres_substorm)

np.save('../output/Bx_nosubstorm'+str(per_substorm), Bx_nosubstorm)
np.save('../output/By_nosubstorm'+str(per_substorm), By_nosubstorm)
np.save('../output/Bz_nosubstorm'+str(per_substorm), Bz_nosubstorm)
np.save('../output/vel_nosubstorm'+str(per_substorm), vel_nosubstorm)
np.save('../output/rho_n_nosubstorm'+str(per_substorm), rho_n_nosubstorm)
np.save('../output/pres_nosubstorm'+str(per_substorm), pres_nosubstorm)
np.save('../output/SML_nosubstorm'+str(per_substorm), SML_nosubstorm)

# comparison between substorms and non-substorms events
length = len(Bz_substorm[0])
Bx_substorm_mean = np.nanmean(Bx_substorm, 0)
By_substorm_mean = np.nanmean(By_substorm, 0)
Bz_substorm_mean = np.nanmean(Bz_substorm, 0)
vel_substorm_mean = np.nanmean(vel_substorm, 0)
rho_n_substorm_mean = np.nanmean(rho_n_substorm, 0)
pres_substorm_mean = np.nanmean(pres_substorm, 0)

Bx_nosubstorm_mean = np.nanmean(Bx_nosubstorm, 0)
By_nosubstorm_mean = np.nanmean(By_nosubstorm, 0)
Bz_nosubstorm_mean = np.nanmean(Bz_nosubstorm, 0)
vel_nosubstorm_mean = np.nanmean(vel_nosubstorm, 0)
rho_n_nosubstorm_mean = np.nanmean(rho_n_nosubstorm, 0)
pres_nosubstorm_mean = np.nanmean(pres_nosubstorm, 0)

figs, axes = plt.subplots(6, 1)
plt.setp(axes, xticks=np.linspace(0, length, 3), xticklabels=[
    '-01:00', '-00:30', '00:00'])
plt.xlabel('Epoch Time(h)', fontsize=20, labelpad=30)

plt.sca(axes[0])
plt.plot(range(length), Bx_substorm_mean, 'g', label='substorm')
plt.plot(range(length), Bx_nosubstorm_mean, 'b', label='non-substorm')
plt.ylabel('IMF Bx/nT', rotation=0, labelpad=40, fontsize=15)
plt.legend(loc='upper right', prop={'size': 20})

plt.sca(axes[1])
plt.plot(range(length), By_substorm_mean, 'g')
plt.plot(range(length), By_nosubstorm_mean, 'b')
plt.ylabel('IMF By/nT', rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[2])
plt.plot(range(length), Bz_substorm_mean, 'g')
plt.plot(range(length), Bz_nosubstorm_mean, 'b')
plt.ylabel('IMF Bz/nT', rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[3])
plt.plot(range(length), vel_substorm_mean, 'g')
plt.plot(range(length), vel_nosubstorm_mean, 'b')
plt.ylabel('Solar Wind \n Speed \n/(km/s)',
           rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[4])
plt.plot(range(length), rho_n_substorm_mean, 'g')
plt.plot(range(length), rho_n_nosubstorm_mean, 'b')
plt.ylabel('Solar Wind \n Number Density \n/(1/cc)',
           rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[5])
plt.plot(range(length), pres_substorm_mean, 'g')
plt.plot(range(length), pres_nosubstorm_mean, 'b')
plt.ylabel('Solar Wind \n Pressure \n/nPa',
           rotation=0, labelpad=40, fontsize=15)

plt.suptitle(
    'comparison between substorms and non-substorms events', fontsize=20)

# assign to the three groups and to matrix and normalize

# three groups number
per_train = 60/100
per_vali = 15/100
per_test = 25/100

num_train = int(num_tot*per_train)
num_vali = int(num_tot*per_vali)
num_test = num_tot-num_vali-num_train

# shuffle the order
odr_cases = list(range(num_tot))  # order number of the casess
random.shuffle(odr_cases)
odr_train = odr_cases[:num_train]
odr_vali = odr_cases[num_train:num_train + num_vali]
odr_test = odr_cases[num_train+num_vali:]

Bx_train = Bx_cases[odr_train]
Bx_vali = Bx_cases[odr_vali]
Bx_test = Bx_cases[odr_test]

By_train = By_cases[odr_train]
By_vali = By_cases[odr_vali]
By_test = By_cases[odr_test]

Bz_train = Bz_cases[odr_train]
Bz_vali = Bz_cases[odr_vali]
Bz_test = Bz_cases[odr_test]

vel_train = vel_cases[odr_train]
vel_vali = vel_cases[odr_vali]
vel_test = vel_cases[odr_test]

rho_n_train = rho_n_cases[odr_train]
rho_n_vali = rho_n_cases[odr_vali]
rho_n_test = rho_n_cases[odr_test]

pres_train = pres_cases[odr_train]
pres_vali = pres_cases[odr_vali]
pres_test = pres_cases[odr_test]

label = np.array([1]*num_substorm+[0]*num_nosubstorm)

label_train = np.array(label)[odr_train]
label_vali = np.array(label)[odr_vali]
label_test = np.array(label)[odr_test]

train_data = np.stack(([normalize(Bx_train, axis=1), normalize(By_train, axis=1), normalize(Bz_train, axis=1), normalize(
    vel_train, axis=1), normalize(rho_n_train, axis=1), normalize(pres_train, axis=1)]), axis=1)
train_data = train_data.reshape(
    [train_data.shape[0], train_data.shape[1], train_data.shape[2], 1])
vali_data = np.stack(([normalize(Bx_vali, axis=1), normalize(By_vali, axis=1), normalize(Bz_vali, axis=1), normalize(
    vel_vali, axis=1), normalize(rho_n_vali, axis=1), normalize(pres_vali, axis=1)]), axis=1)
vali_data = vali_data.reshape(
    (vali_data.shape[0], vali_data.shape[1], vali_data.shape[2], 1))
test_data = np.stack(([normalize(Bx_test, axis=1), normalize(By_test, axis=1), normalize(Bz_test, axis=1), normalize(
    vel_test, axis=1), normalize(rho_n_test, axis=1), normalize(pres_test, axis=1)]), axis=1)
test_data = test_data.reshape(
    (test_data.shape[0], test_data.shape[1], test_data.shape[2], 1))

# save data
np.save("../output/train_data.npy"+str(per_substorm), train_data)
np.save("../output/vali_data.npy"+str(per_substorm), vali_data)
np.save("../output/test_data.npy"+str(per_substorm), test_data)
np.save("../output/label_train.npy"+str(per_substorm), label_train)
np.save("../output/label_test.npy"+str(per_substorm), label_test)
np.save("../output/label_vali.npy"+str(per_substorm), label_vali)

# %% model
# model build
model_f = 1
if model_f == 1:
    model = models.Sequential()
    model.add(layers.Conv2D(
        128, (6, 3), activation='relu', input_shape=(6, 60, 1)))
    model.add(layers.MaxPooling2D((1, 3)))
    model.add(layers.Conv2D(256, (1, 3), activation='relu'))
    model.add(layers.MaxPooling2D((1, 3)))
    model.add(layers.Conv2D(512, (1, 3), activation='relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(32))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    # model compile
    adam = tf.keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data, label_train, epochs=10, batch_size=64,
                        validation_data=(vali_data, label_vali))


elif model_f == 2:  # LSTM
    model = tf.keras.Sequential()

    train_data = train_data.reshape(-1, 360)
    vali_data = vali_data.reshape(-1, 360)
    test_data = test_data.reshape(-1, 360)
    model.add(layers.Embedding(input_dim=360, output_dim=64))
    model.add(layers.LSTM(128))
    model.add(layers.Dense(1))
    model.summary()

    # model compile
    adam = tf.keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data, label_train, epochs=10, batch_size=64,
                        validation_data=(vali_data, label_vali))

# plot
plt.figure()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='lower right')
title_dic = {1: 'CNN', 2: 'LSTM'}
plt.title(title_dic[model_f] +
          ' accuracy and loss for train and validation sets')
plt.show()

if model_f==1:
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, label_train, epochs=3, batch_size=64,
                        validation_data=(vali_data, label_vali))
else:
    model.compile(optimizer=adam,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_data, label_train, epochs=10, batch_size=64,
                        validation_data=(vali_data, label_vali))  
test_loss, test_acc = model.evaluate(test_data,  label_test, verbose=2)

print('The test accuracy is %f.' % test_acc)
# %% permutation importance


def importance():
    def score(X, y):
        y_pred = model(X)
        y_pred = np.where(y_pred > 0.5, 1, 0).flatten()
        return accuracy_score(y, y_pred)

    base_score, score_decreases = get_score_importances(
        score, test_data, label_test)
    feature_importances = np.mean(score_decreases, axis=0)
    feature_importances = feature_importances / \
        np.linalg.norm(feature_importances)

    return feature_importances


feature_importances = importance()
print('The feature importances for solar wind quantitites are: IMF Bx: %f, IMF By: %f, IMF Bz: %f, solar wind velocity: %f\
      solar wind number density: %f, solar wind pressure: %f' % (feature_importances[0], feature_importances[1], feature_importances[2],
                                                                 feature_importances[3], feature_importances[4], feature_importances[5],))
# %% case study
case_id = 500
starttime_case = time_substorm[case_id]
starttime_id = time.index(starttime_case)
period_id = range(starttime_id-180, starttime_id+180)
Bx_case_study = Bx[period_id]
By_case_study = By[period_id]
Bz_case_study = Bz[period_id]
vel_case_study = vel[period_id]
rho_n_case_study = rho_n[period_id]
pres_case_study = pres[period_id]
SML_case_study = SML[period_id]

length = len(period_id)
figs, axes = plt.subplots(7, 1, figsize=(20, 15))
plt.setp(axes, xticks=np.linspace(0, length, 7), xticklabels=[
    '00:15', '01:15', '02:15', '03:15', '04:15', '05:15', '06:15'])
plt.xlabel('Universal Time(h)', fontsize=20, labelpad=20)
plt.sca(axes[0])
plt.plot(range(length), Bx_case_study)
plt.ylabel('IMF Bx/nT', rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[1])
plt.plot(range(length), By_case_study)
plt.ylabel('IMF By/nT', rotation=0, labelpad=40, fontsize=15)

plt.sca(axes[2])
plt.plot(range(length), Bz_case_study)
plt.ylabel('IMF Bz/nT', rotation=0, labelpad=30, fontsize=15)

plt.sca(axes[3])
plt.plot(range(length), vel_case_study)
plt.ylabel('Solar Wind Speed \n (km/s)', rotation=0, labelpad=80, fontsize=15)

plt.sca(axes[4])
plt.plot(range(length), rho_n_case_study)
plt.ylabel('Solar Wind \n Number Density\n (n/cc)',
           rotation=0, labelpad=80, fontsize=15)

plt.sca(axes[5])
plt.plot(range(length), pres_case_study)
plt.ylabel('Solar Wind Pressure\n (nPa)', rotation=0, labelpad=85, fontsize=15)

plt.sca(axes[6])
plt.plot(range(length), SML_case_study)
plt.ylabel('SML\n (nT)', rotation=0, labelpad=30, fontsize=15)

plt.suptitle('Case study: 1996 December 15th', fontsize=20)
