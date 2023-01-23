# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 07:35:05 2022

@author: natha
"""

#import necessary modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#defined attributes for later code
#define file paths for excel files for easier edits if needed
path_defense = r'C:\Users\natha\Documents\School\cis 450\indiviudal project\Cubs_all_time_stats_defense.xlsx'
path_offense = r'C:\Users\natha\Documents\School\cis 450\indiviudal project\Cubs_all_time_stats_offense.xlsx'
#end of file paths


#define heat map axis
def_heat = "Defensive Stat features"
off_heat = "Offensive Stat features"
#end of heat map axis

#define functions for later in code
#print function for top 5 correlations later
def print_function(text, correlation):
  print(text)
  print(correlation, '\n')
  
#print function for the first pearson correlations and p-values 
def stat_print_function_header(text, text2, stats):
    print(text)
    print('\n', text2)
    print(stats, '\n')

#print function for pearson and pvalues after the header has one less text
def stat_print_function(text, stats):
    print(text)
    print(stats, '\n')
    
#setup heatmap function
def create_heatmap(input_corr_mat, title, xlabel, ylabel):
    sns.heatmap(input_corr_mat, annot = True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
def r_squared_print(text, score):
    print(text)
    print(score, '\n')
    
def coefficent_function(text, coefficent):
    print(text)
    print(coefficent)
    
def linear_function(text, text1, intercept, text2, coefficient, text3, rsquared, text4, meansquarederror):
  print(text, '\n')
  print(text1)
  print(intercept, '\n')
  print(text2)
  print(coefficient, '\n')
  print(text3)
  print(rsquared, '\n')
  print(text4)
  print(meansquarederror, '\n')
  
#end of defined functions
#end of defined attributes
print()
  
#read excel file for correlations later
df2 = defense_1 = pd.read_excel(path_defense, sheet_name='Worksheet')

#first defense sheet for appropriate sized heat map
defense_1 = pd.read_excel(path_defense, sheet_name='defense1')
correlation_mat = defense_1.corr()
create_heatmap(correlation_mat, "Correlation matrix of Defensive Stats 1", def_heat, def_heat)


#read excel file 
#demonstrate building dataframe in python for the desired columns
defense_2 = pd.read_excel(path_defense, sheet_name='All Stats for dataframe', usecols="A:D , K:Q")
#this would be the same as using sheet defense 2 from the excel file
correlation_mat2 = defense_2.corr()
create_heatmap(correlation_mat2, "Correlation matrix of Defensive Stats 2", def_heat, def_heat)

#third defense sheet for appropriate sized heat map
defense_3 = pd.read_excel(path_defense, sheet_name='defense3')
correlation_mat3 = defense_3.corr()
create_heatmap(correlation_mat3, "Correlation matrix of Defensive Stats 3" , def_heat, def_heat)

#fourth defense sheet for appropriate sized heat map
defense_4 = pd.read_excel(path_defense, sheet_name='defense4')
correlation_mat4 = defense_4.corr()
create_heatmap(correlation_mat4, "Correlation matrix of Defensive Stats 4" , def_heat, def_heat)

#first offense sheet for appropriate sized heat map
offense_1 = pd.read_excel(path_offense, sheet_name='offense1')
correlation_mat5 = offense_1.corr()
create_heatmap(correlation_mat5, "Correlation matrix of Offensive Stats 1", off_heat, off_heat)

#second offense sheet for appropriate sized heat map
offense_2 = pd.read_excel(path_offense, sheet_name='offense2')
correlation_mat6 = offense_2.corr()
create_heatmap(correlation_mat6, "Correlation matrix of Offensive Stats 2", off_heat, off_heat)

#third offense sheet for appropriate sized heat map
offense_3 = pd.read_excel(path_offense, sheet_name='offense3')
correlation_mat7 = offense_3.corr()
create_heatmap(correlation_mat7, "Correlation matrix of Offensive Stats 3", off_heat, off_heat)

#fourth offense sheet for appropriate sized heat map
offense_4 = pd.read_excel(path_offense, sheet_name='offense4')
correlation_mat8 = offense_4.corr()
create_heatmap(correlation_mat8, "Correlation matrix of Offensive Stats 4", off_heat, off_heat)


#read our excel file sheet with win loss ratio
df2WL = pd.read_excel(path_defense, sheet_name='Worksheet_WL')

#find and sort correlations related to our def W variable (wins)
win_correlations = df2[df2.columns[0:]].corr()['W'].abs().sort_values(ascending=False).iloc[1:6]
#iloc 1-6 as the first value will be 1.000 as it will be the variable we are comparing so we do not want that value
print_function("Top 5 Defensive Wins Absolute Correlations:", win_correlations)

#find and sort correlations related to our def L variable (Loss)
#reference df2 for our entire data set
loss_correlations = df2[df2.columns[0:]].corr()['L'].abs().sort_values(ascending=False).iloc[1:6]
#iloc 1-6 as the first value will be 1.000 as it will be the variable we are comparing so we do not want that value
print_function("Top 5 Defensive Loss Absoulte Correlations:", loss_correlations)

#find and sort correlations related to our def W/L Ratio variable (Win Loss Ratio)
#reference df2 for our entire data set
win_loss_correlations = df2WL[df2WL.columns[0:]].corr()['W/L Ratio'].abs().sort_values(ascending=False).iloc[1:6]
#iloc 1-6 as the first value will be 1.000 as it will be the variable we are comparing so we do not want that value
print_function("Top 5 Defensive Win/Loss Ratio Absolute Correlations:", win_loss_correlations)





#read our excel file for offensive stats for our wins and losses variables
df4 = pd.read_excel(path_offense, sheet_name='Worksheet')

#read our excel file sheet with win loss ratio
df4WL = pd.read_excel(path_offense, sheet_name='Worksheet_WL')

#find and sort correlations related to our def W variable (wins)
#reference df2 for our entire data set
win_correlations = df4[df4.columns[0:]].corr()['W'].abs().sort_values(ascending=False).iloc[1:6]
#iloc 1-6 as the first value will be 1.000 as it will be the variable we are comparing so we do not want that value
print_function("Top 5 Offensive Wins Absolute Correlations:", win_correlations)

#find and sort correlations related to our def L variable (Loss)
#reference df2 for our entire data set
loss_correlations = df4[df4.columns[0:]].corr()['L'].abs().sort_values(ascending=False).iloc[1:6]
#iloc 1-6 as the first value will be 1.000 as it will be the variable we are comparing so we do not want that value
print_function("Top 5 Offensive Loss Absoulte Correlations:", loss_correlations)

#find and sort correlations related to our def W/L Ratio variable (Win Loss Ratio)
#reference df2 for our entire data set
win_loss_correlations = df4WL[df4WL.columns[0:]].corr()['W/L Ratio'].abs().sort_values(ascending=False).iloc[1:6]
#iloc 1-6 as the first value will be 1.000 as it will be the variable we are comparing so we do not want that value
print_function("Top 5 Offensive Win/Loss Ratio Absolute Correlations:", win_loss_correlations)

#begin testing for defensive statistical significance 
#defense win 1 abbreviated as dw1, dw2, dw3 etc 
#run test for highest correlation for defensive wins
dw1 = pearsonr(df2['W'], df2['tSho'])
stat_print_function_header("The correlations and p-value for defensive wins are listed below:", \
                           "The pearson correlation and p-value for defensive wins and shutouts respectively are:",\
                               dw1)


#run test for 2nd highest correlation for defensive wins
dw2 = pearsonr(df2['W'], df2['IP'])
stat_print_function("The pearson correlation and p-value for defensive wins and innings pitched respectively are:",\
                    dw2)


#run test for 3rd highest correlation for defensive wins
dw3 = pearsonr(df2['W'], df2['RA/G'])
stat_print_function("The pearson correlation and p-value for defensive wins and runs allowed per game respectively are:",\
                    dw3)


#run test for 4th highest correlation for defensive wins
dw4 = pearsonr(df2['W'], df2['Finish'])
stat_print_function("The pearson correlation and p-value for defensive wins and placement respectively are:",\
                    dw4)


#run test for 5th highest correlation for defensive wins
dw5 = pearsonr(df2['W'], df2['G'])
stat_print_function("The pearson correlation and p-value for defensive wins and games played respectively are:",\
                    dw5)


#defense loss 1 abbreviated as dl1, dl2, dl3 etc 
#run test for highest correlation for defensive losses
dl1 = pearsonr(df2['L'], df2['ER'])
stat_print_function_header("The correlations and p-value for defensive losses are listed below:", \
                           "The pearson correlation and p-value for defensive losses and earned runs respectively are:",\
                               dl1)

#run test for 2nd highest correlation for defensive losses
dl2 = pearsonr(df2['L'], df2['H'])
stat_print_function("The pearson correlation and p-value for defensive losses and hits allowed respectively are:",\
                    dl2)


#run test for 3rd highest correlation for defensive losses
dl3 = pearsonr(df2['L'], df2['BB'])
stat_print_function("The pearson correlation and p-value for defensive losses and walks allowed respectively are:",\
                    dl3)


#run test for 4th highest correlation for defensive losses
dl4 = pearsonr(df2['L'], df2['G'])
stat_print_function("The pearson correlation and p-value for defensive losses and games played respectively are:",\
                    dl4)


#run test for 5th highest correlation for defensive losses
dl5 = pearsonr(df2['L'], df2['R'])
stat_print_function("The pearson correlation and p-value for defensive losses and runs allowed per game respectively are:",\
                    dl5)


#defense win-loss 1 abbreviated as dwl1, dwl2, dwl3 etc 
#run test for highest correlation for defensive win-loss ratio
dwl1 = pearsonr(df2WL['W/L Ratio'], df2WL['Finish'])
stat_print_function_header("The correlations and p-value for defensive win-loss ratio are listed below:", \
                           "The pearson correlation and p-value for defensive win-loss ratio and placement respectively are:",\
                               dwl1)


#run test for 2nd highest correlation for defensive win-loss ratio
dwl2 = pearsonr(df2WL['W/L Ratio'], df2WL['ERA'])
stat_print_function("The pearson correlation and p-value for defensive win-loss ratio and ERA respectively are:",\
                    dwl2)


#run test for 3rd highest correlation for defensive win-loss ratio
dwl3 = pearsonr(df2WL['W/L Ratio'], df2WL['ER'])
stat_print_function("The pearson correlation and p-value for defensive win-loss ratio and runs allowed respectively are:",\
                    dwl3)

#run test for 4th highest correlation for defensive win-loss ratio
dwl4 = pearsonr(df2WL['W/L Ratio'], df2WL['WHIP'])
stat_print_function("The pearson correlation and p-value for defensive win-loss ratios and WHIP respectively are:",\
                    dwl4)


#run test for 5th highest correlation for defensive win-loss ratio
dwl5 = pearsonr(df2WL['W/L Ratio'], df2WL['R'])
stat_print_function("The pearson correlation and p-value for defensive win-loss ratio and runs allowed respectively are:",\
                    dwl5)


#begin testing for offensive statistical significance 
#offense win 1 abbreviated as ow1, ow2, ow3 etc 
#run test for highest correlation for offensive wins
ow1 = pearsonr(df4['W'], df4['PA'])
stat_print_function_header("The correlations and p-value for offensive wins are listed below:", \
                           "The pearson correlation and p-value for offensive wins and plate appearences respectively are:",\
                               ow1)


#run test for 2nd highest correlation for offensive wins
ow2 = pearsonr(df4['W'], df4['Finish'])
stat_print_function("The pearson correlation and p-value for offensive wins and placement respectively are:",\
                    ow2)


#run test for 3rd highest correlation for offensive wins
ow3 = pearsonr(df4['W'], df4['G'])
stat_print_function("The pearson correlation and p-value for offensive wins and games played respectively are:",\
                    ow3)


#run test for 4th highest correlation for offensive wins
ow4 = pearsonr(df4['W'], df4['H'])
stat_print_function("The pearson correlation and p-value for offensive wins and hits respectively are:",\
                    ow4)

#run test for 5th highest correlation for offensive wins
ow5 = pearsonr(df4['W'], df4['AB'])
stat_print_function("The pearson correlation and p-value for offensive wins and at-bats respectively are:",\
                    ow5)



#offense loss 1 abbreviated as ol1, ol2, ol3 etc 
#run test for highest correlation for offensive losses
ol1 = pearsonr(df4['L'], df4['AB'])
stat_print_function_header("The correlations and p-value for offensive losses are listed below:",\
                           "The pearson correlation and p-value for offensive losses and at-bats respectively are:",\
                               ol1)


#run test for 2nd highest correlation for offensive losses
ol2 = pearsonr(df4['L'], df4['G'])
stat_print_function("The pearson correlation and p-value for offensive losses and games played respectively are:",\
                    ol2)

#run test for 3rd highest correlation for offensive losses
ol3 = pearsonr(df4['L'], df4['PA'])
stat_print_function("The pearson correlation and p-value for offensive losses and plate appearences respectively are:",\
                    ol3)

#run test for 4th highest correlation for offensive losses
ol4 = pearsonr(df4['L'], df4['DP'])
stat_print_function("The pearson correlation and p-value for offensive losses and double plays turned respectively are:",\
                    ol4)


#run test for 5th highest correlation for offensive losses
ol5 = pearsonr(df4['L'], df4['Finish'])
stat_print_function("The pearson correlation and p-value for offensive losses and placement respectively are:",\
                    ol5)



#offense win-loss 1 abbreviated as owl1, owl2, owl3 etc 
#run test for highest correlation for offensive win-loss ratio
owl1 = pearsonr(df4WL['W/L Ratio'], df4WL['Finish'])
stat_print_function_header("The correlations and p-value for offensive win-loss ratio are listed below:", \
                           "The pearson correlation and p-value for offensive win-loss ratio and placement respectively are:",\
                               owl1)


#run test for 2nd highest correlation for offensive win-loss ratio
owl2 = pearsonr(df4WL['W/L Ratio'], df4WL['R/G'])
stat_print_function("The pearson correlation and p-value for offensive win-loss ratio and runs per game respectively are:",\
                    owl2)


#run test for 3rd highest correlation for offensive win-loss ratio
owl3 = pearsonr(df4WL['W/L Ratio'], df4WL['DP'])
stat_print_function("The pearson correlation and p-value for offensive win-loss ratio and double plays turned respectively are:",\
                    owl3)


#run test for 4th highest correlation for offensive win-loss ratio
owl4 = pearsonr(df4WL['W/L Ratio'], df4WL['BA'])
stat_print_function("The pearson correlation and p-value for offensive win-loss ratios and hits per at bat respectively are:",\
                    owl4)


#run test for 5th highest correlation for offensive win-loss ratio
owl5 = pearsonr(df4WL['W/L Ratio'], df4WL['SO'])
stat_print_function("The pearson correlation and p-value for offensive win-loss ratio and strikeouts respectively are:",\
                    owl5)
    
    
#run regression analysis on all variables
#defensive wins
dfx = df2[['Finish', 'RA/G', 'ERA', 'G', 'CG', 'tSho', 'SV', 'IP', 'H', 'R', 'ER', 'HR', 'BB', 'SO', 'WHIP', 'SO9', 'E',\
         'DP', 'Fld%', 'PAge', 'WPG', 'HRPG', 'EPG', 'HPG', 'SV%', 'CG%']]
dfwy = df2['W']

mlr_dfw = LinearRegression() 
mlr_dfw.fit(dfx,dfwy)

dfw_pred = mlr_dfw.predict(dfx)   
r2_score_dfw = round(mlr_dfw.score(dfx.values,dfwy),2)
    
linear_function("Linear Regression stats for defensive wins: ", "Intercept: ", mlr_dfw.intercept_, "Coefficients:",\
                list(zip(dfx, mlr_dfw.coef_)), "R squared score of regression: ", r2_score_dfw, "RMSE:",\
                    np.sqrt(mean_squared_error(dfwy,dfw_pred)))
    
#defensive losses
dfly = df2['L']

mlr_dfl = LinearRegression() 
mlr_dfl.fit(dfx,dfly)

dfl_pred = mlr_dfl.predict(dfx)   
r2_score_dfl = round(mlr_dfl.score(dfx.values,dfly),2)
    
linear_function("Linear Regression stats for defensive losses: ","Intercept: ", mlr_dfl.intercept_, "Coefficients:",\
                list(zip(dfx, mlr_dfl.coef_)), "R squared score of regression: ", r2_score_dfl, "RMSE:",\
                    np.sqrt(mean_squared_error(dfly,dfl_pred)))
    
#offensive wins
ofx = df4[['Finish', 'R/G', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'BB', 'SO', 'BA', 'OBP',\
         'SLG', 'OPS', 'E', 'DP', 'Fld%', 'BatAge', '2B%', '3B%', 'HR%', 'SO%', 'BB%', 'DPPG']]
ofwy = df4['W']

mlr_ofw = LinearRegression() 
mlr_ofw.fit(ofx,ofwy)

ofw_pred = mlr_ofw.predict(ofx)   
r2_score_ofw = round(mlr_ofw.score(ofx.values,ofwy),2)
    
linear_function("Linear Regression stats for offensive wins: ", "Intercept: ", mlr_ofw.intercept_, "Coefficients:",\
                list(zip(ofx, mlr_ofw.coef_)), "R squared score of regression: ", r2_score_ofw, "RMSE:",\
                    np.sqrt(mean_squared_error(ofwy,ofw_pred)))    
        
    
#offensive losses
ofly = df2['L']

mlr_ofl = LinearRegression() 
mlr_ofl.fit(ofx,ofly)

ofl_pred = mlr_ofl.predict(ofx)   
r2_score_ofl = round(mlr_ofl.score(ofx.values,ofly),2)
    
linear_function("Linear Regression stats for offensive losses: ","Intercept: ", mlr_ofl.intercept_, "Coefficients:",\
                list(zip(ofx, mlr_ofl.coef_)), "R squared score of regression: ", r2_score_ofl, "RMSE:",\
                    np.sqrt(mean_squared_error(ofly,ofl_pred)))    


    
#build predicitve regression model
#regression called in each regression test

regr = linear_model.LinearRegression()

#start with defensive characteristics
#defensive wins regression and prediction
def_w_x = df2[['tSho', 'IP', 'RA/G', 'Finish', 'G']]
def_w_y = df2['W']

regr.fit(def_w_x.values,def_w_y)

shutous = 18
innings_pitched = 1463.8
runs_allowed_per_game = 4.86
finish = 2
games_played = 162

predicted_wins_defense = regr.predict([[shutous, innings_pitched, runs_allowed_per_game, finish, games_played]])

print("Predicted wins with",shutous,"shutous,", innings_pitched,"innings pitched,", runs_allowed_per_game, "runs allowed per game,", 
      finish,"place finish, and", games_played, "games played are:",predicted_wins_defense)

r2_score_def_w = round(regr.score(def_w_x.values,def_w_y),2)
r_squared_print("R squared score of regression: ", r2_score_def_w)
coefficent_function("Coefficents: ", list(zip(def_w_x, regr.coef_)))

#defensive losses regression and prediction
def_l_x = df2[['ER', 'H', 'BB', 'G', 'R']]
def_l_y = df2['L']

regr.fit(def_l_x.values,def_l_y)



earned_runs = 292
hits_allowed = 998
walks_allowed = 512
games_played = 163
runs_allowed = 656

predicted_losses_defense = regr.predict([[earned_runs, hits_allowed, walks_allowed, games_played, runs_allowed]])

print("Predicted Losses with",earned_runs,"earned runs,", hits_allowed,"hits_allowed", walks_allowed,"walks allowed are:",\
      games_played, "games played, and", runs_allowed, "runs allowed are:",predicted_losses_defense)
    
r2_score_def_l = round(regr.score(def_l_x.values,def_l_y),2)
r_squared_print("R squared score of regression: ", r2_score_def_l)
coefficent_function("Coefficents: ", list(zip(def_l_x, regr.coef_)))



#prediction for offensive stats
#offensive wins regression and prediction

off_w_x = df4[['PA', 'Finish', 'G', 'H', 'AB']]
off_w_y = df4['W']

regr.fit(off_w_x.values,off_w_y)

plate_appearances = 6345
finish = 3
games_played = 161
hits = 1496
at_bats = 5422

predicted_wins_offense = regr.predict([[plate_appearances, finish, games_played, hits, at_bats]])

print("Predicted wins with",plate_appearances,"plate appearances,", finish,"place finish,", games_played,"games played,",\
      hits, "hits, and", at_bats, "at bats are:",predicted_wins_offense)
    
r2_score_off_w = round(regr.score(off_w_x.values,off_w_y),2)
r_squared_print("R squared score of regression: ", r2_score_off_w)
coefficent_function("Coefficents: ", list(zip(off_w_x, regr.coef_)))

#offensive losses regression and prediction
off_l_x = df4[['AB', 'G', 'PA', "DP", "Finish"]]
off_l_y = df4['L']

regr.fit(off_l_x.values,off_l_y)


at_bats = 5312
games_played = 163
plate_appearances = 6112
double_plays = 126
finish = 1

predicted_losses_offense = regr.predict([[at_bats, games_played, plate_appearances, double_plays, finish]])
print("Predicted losses with",at_bats,"at bats,", games_played,"games played,", plate_appearances,"plate appearances,",\
      double_plays, "double plays, and", finish, "place finish are:",predicted_losses_offense)
    
r2_score_off_l = round(regr.score(off_l_x.values,off_l_y),2)
r_squared_print("R squared score of regression: ", r2_score_off_l)
coefficent_function("Coefficents: ", list(zip(off_l_x, regr.coef_)))