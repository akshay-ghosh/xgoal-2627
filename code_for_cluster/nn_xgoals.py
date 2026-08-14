# nn_xgoals.py
########################################################################
# IMPORTS

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # for non-gui
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.calibration import calibration_curve

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from scipy.stats import describe
from scipy.stats import linregress

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_curve, roc_auc_score, precision_recall_curve

# from sklearn.model_selection import GridSearchCV

# from sklearn.base import BaseEstimator, TransformerMixin

# import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

# from imblearn.under_sampling import RandomsUnderSampler

# # plot the rink
# from hockey_rink import NHLRink, RinkImage

########################################################################
# LOAD DATA

plot_folder = '/Users/akshayghosh/hockey/xgoal_hpc/cluster_plots/'

# each season of shooting data for training
shots_dir = '/Users/akshayghosh/hockey/expected_goals_model/shot_data/' # local
# shots_dir = '/home/aghosh/projects/def-aghosh/aghosh/ncloud/data/xgoal_shot_data/' # narval

fn_shots = [shots_dir + 'shots_2017_2018.csv',
            shots_dir + 'shots_2018_2019.csv',
            shots_dir + 'shots_2019_2020.csv',
            shots_dir + 'shots_2020_2021.csv', 
            shots_dir + 'shots_2021_2022.csv',
            shots_dir + 'shots_2022_2023.csv']


# fn_shots = ['/home/aghosh/projects/def-aghosh/aghosh/ncloud/data/xgoal_shot_data/shots_2017_2018.csv',
#             '/home/aghosh/projects/def-aghosh/aghosh/ncloud/data/xgoal_shot_data/shots_2018_2019.csv',
#             '/home/aghosh/projects/def-aghosh/aghosh/ncloud/data/xgoal_shot_data/shots_2019_2020.csv',
#             '/home/aghosh/projects/def-aghosh/aghosh/ncloud/data/xgoal_shot_data/shots_2020_2021.csv', 
#             '/home/aghosh/projects/def-aghosh/aghosh/ncloud/data/xgoal_shot_data/shots_2021_2022.csv',
#             '/home/aghosh/projects/def-aghosh/aghosh/ncloud/data/xgoal_shot_data/shots_2022_2023.csv']
# fn_shots = ['./shot_data/shots_2020_2021.csv', 
#             './shot_data/shots_2021_2022.csv', 
#             './shot_data/shots_2022_2023.csv']

# read each CSV file into a DataFrame and concatenate them
data = pd.concat([pd.read_csv(file) for file in fn_shots], ignore_index=True)
data = data[data['isPlayoffGame'] == 0]
# data = data[data['shotOnEmptyNet'] == 0]
# data = data[~data['lastEventCategory'].isin(['PEND', 'GEND', 'EGT'])] # USED THIS WHEN THE TRAINING DATA WAS 20-21 TO 22-23
data = data[~data['lastEventCategory'].isin(['PENL','STOP','GOAL','CHL','PEND','PSTR','ANTHEM','EISTR', 'GEND', 'EGT'])] # try this when training data is extended to be 17-18 to 22-23

# make predictions on last year of data
fn_predict = shots_dir + 'shots_2023_2024.csv'
data_predict = pd.read_csv(fn_predict)
data_predict = data_predict[data_predict['isPlayoffGame'] == 0]
data_predict = data_predict[~data_predict['lastEventCategory'].isin(['PENL','STOP','GOAL','CHL','PEND','PSTR','ANTHEM','EISTR', 'GEND', 'EGT'])] # try this when training data is extended to be 17-18 to 22-23
# data_predict = data_predict[data_predict['shotOnEmptyNet'] == 0]

########################################################################
# SELECT FEATURES

y = data['goal']

features = ['timeSinceLastEvent', # Time between the shot and the event that took place before the shot
                 'shotAngle', # The angle of the shot in degrees. Is a positive number if the shot is from the left side of the ice. 
                 'shotAnglePlusRebound',# The angle of the shot in degrees. Is a positive number if the shot is from the left side of the ice. 
                 'shotAngleReboundRoyalRoad', # Set to 1 if the puck went through the middle of the between this shot and previous shot if this shot is a rebound.
                 'shotDistance', # The distance from the net of the shot in feet. Net is defined as being at the (89,0) coordinates
                 'shotType', # Type of the shot. (Slap, Wrist, etc)
                 'shotRebound', # Set to 1 if the shot is a rebound. (If the last event was a shot and within 3 seconds of this shot)
                 'shotAnglePlusReboundSpeed', # The shotAnglePlusRebound variable divded by time between the last shot and this one. (How fast the angle changed)
                #  'shotRush', # Set to 1 if the shot was on a rush. (If the last event was in another zone and within 4 seconds)
                 'speedFromLastEvent', # The distance between the shot location and the previous event's location divded by the number of seconds between them
                 'distanceFromLastEvent', # The distance between the shot location and the previous event's location in feet
                 'lastEventShotAngle', # The shot angle of the shot directly before this shot. (If the last event was a shot)
                 'lastEventShotDistance', # The shot distance of the shot directly before this shot. (If the last event was a shot)
                 'lastEventCategory', # The type of event before the shot.Shot, hit, etc.
                 'shooterTimeOnIce', # playing time in seconds that have passed since the shooter started their shift
                 'shootingTeamAverageTimeOnIce', # The average playing time in seconds the shooting team's players have been on the ice
                 'defendingTeamAverageTimeOnIce', # The average playing time in seconds the shooting team's players have been on the ice
                 'offWing', # Set to 1 if the shot is from the left side of the ice and the shooter is a right shot, or vice-versa. Otherwise 0
                 'shotOnEmptyNet'] # set to 0 if the net is empty, 1 if non-empty


print('Number of features: ',len(features))

X = data[features]
X_predict = data_predict[features]

p_num = X.select_dtypes(include=['int64', 'float64'])
p_cat = X.select_dtypes(include=['object'])

# print(f'numerical features:\n{p_num.columns}\n')
# print(f'categorical features:\n{p_cat.columns}')

# define transformations
preprocessor = ColumnTransformer([
    ('cat_encoder', OneHotEncoder(handle_unknown="ignore"), p_cat.columns)  # one-hot encode categorical features
], remainder="passthrough")  # ensure binary features remain unchanged

########################################################################
# FIT DATA

# fit-transform data
X_transformed = preprocessor.fit_transform(X)

# get feature names
feature_names = preprocessor.get_feature_names_out()

# convert to DataFrame
X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

X_postprocess = preprocessor.fit_transform(X)
X_predict_postprocess = preprocessor.fit_transform(X_predict)

X_postprocess_df = pd.DataFrame(X_postprocess, columns=preprocessor.get_feature_names_out())
X_predict_postprocess_df = pd.DataFrame(X_predict_postprocess, columns=preprocessor.get_feature_names_out())

# plt.figure(figsize=(14, 10))
# sns.heatmap(X_postprocess_df.corr(), 
#             annot=True, cmap='viridis', fmt='.2f',
#             annot_kws={"size": 8},  # Adjust annotation font size
#             linewidths=0.5, linecolor='gray')  # Add gridlines for spacing
# plt.title('Feature Correlation Matrix', fontsize=16)  # Increase title font size
# plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for readability
# plt.yticks(fontsize=10)  # Adjust y-axis label font size
# plt.show()

corr_threshold = 0.9
corr_matrix = X_postprocess_df.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]

# [expression for item in iterable if condition]

# drop correlated features
X_postprocess_df = X_postprocess_df.drop(columns=to_drop)
X_predict_postprocess_df = X_predict_postprocess_df.drop(columns=to_drop)
print('Correlated features to drop: ',to_drop)


# up to "Feature correlation matrix"
# actually left out feature correlation matrix for now

# now do low variance features
var_thresh = VarianceThreshold(threshold=0.005)#0.005
X_postprocess_filtered = var_thresh.fit_transform(X_postprocess_df)
selected_columns = X_postprocess_df.columns[var_thresh.get_support()]
low_var = set(X_postprocess_df.columns) - set(selected_columns)
# print('Low variance features: ')
# for feat in low_var:
#     print(feat)

X_postprocess_filtered = pd.DataFrame(X_postprocess_filtered,columns = selected_columns)
X_predict_postprocess_df = pd.DataFrame(X_predict_postprocess_df,columns = selected_columns)

########################################################################
# SPLIT DATA

# split dataset into training and testing
X_train, X_val, y_train, y_val = train_test_split(X_postprocess_filtered.to_numpy(),
                                                  y,
                                                  test_size=0.2,
                                                  random_state=1997,
                                                  stratify=y,
                                                  shuffle = True)
# stratify = y will ensure that the distribution of goals to no goals is the same in the train and val sets


# need X and y to be pytorch tensors
X_train_tensor = torch.FloatTensor(X_train)

y_train_tensor = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val.to_numpy())

# convert validation data to tensors
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = y_val.unsqueeze(1).float()

########################################################################
# DEFINE NEURAL NETWORK

class XGoalNeuralNetwork(nn.Module):

    def __init__(self, in_features=30, h1=512, h2=256, h3=128, h4=64, out_features=1):
        super().__init__()

        # define layers
        self.fc1 = nn.Linear(in_features, h1)
        self.bn1 = nn.BatchNorm1d(h1)  # Batch Normalization
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, out_features)

        # dropout layers with a moderate rate
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # first layer with activation and dropout
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)  # Leaky ReLU activation
        x = self.dropout(x)

        # second layer with activation and dropout
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)

        # third layer with activation and dropout
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)

        # fourth layer
        x = F.relu(self.fc4(x))

        # final output layer (no activation here, as it's regression)
        x = self.out(x)
        # if self.training == False:
        #     T = 1.0 # temperature in final layer, T > 1 reduces overconfidence and lowers probabilities
        #     x = torch.sigmoid(self.out(x) / T)
        # elif self.training == True:
        #     x = self.out(x)
        return x
    
########################################################################
# TRAIN THE MODEL

epochs = 1000
losses = []  # store training losses
losses_val = []  # store validation losses

### INIT MODEL ###
torch.manual_seed(1997)
input_dim = X_train_tensor.shape[1]
# model = GoalPredictionNN(input_dim)
model = XGoalNeuralNetwork()
# model = GoalNetwork()

# criterion = nn.BCELoss(weight=sample_weights)
# criterion = nn.BCELoss()
target_lr = 1e-3
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = target_lr, weight_decay=1e-4)

### LEARNING RATE DYNAMIC ###

# Initial and target learning rates
warmup_epochs = 20
warmup_start_lr = 1e-6  # very small initial LR
# target_lr = 1e-3        # your original LR

# Set initial LR to warmup_start_lr
for param_group in optimizer.param_groups:
    param_group['lr'] = warmup_start_lr

# set up scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# set up early stopping
patience = 30  # stop training if no improvement for 'patience' epochs
best_val_loss = float('inf')  # initialize with a very high value
counter = 0  # count epochs without improvement

for i in range(epochs):
    ### TRAINING PHASE ###
    model.train() # set model to training mode
    optimizer.zero_grad() # gradients need to be set to zero so that they are not accumulated from the previous iteration

    # forward pass on training data
    y_train_calc = model(X_train_tensor)

    # measure loss for training data
    loss_train = criterion(y_train_calc,y_train_tensor.unsqueeze(1).float()) # predicted values vs the y_train
    losses.append(loss_train.item()) # keep track of training losses


    #  back pass (aka backpropagation), update weights
    loss_train.backward()
    optimizer.step()

    ### VALIDATION PHASE ###
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():  # disable gradient calculation during validation
        y_val_calc = model(X_val_tensor)  # forward pass (validation data)
        loss_test = criterion(y_val_calc, y_val_tensor)  # calculate validation loss
        
    losses_val.append(loss_test.item())  # store validation loss
    
    # print every 10 epochs
    if i % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch {i:4d} | Training Loss: {loss_train.item():.4f} | Validation Loss: {loss_test.item():.4f} | log Learning Rate: {np.log10(current_lr):.4f}")
    
    # check early stopping condition
    if loss_test.item() < best_val_loss:
        best_val_loss = loss_test.item()  # Update best loss
        counter = 0  # Reset counter
        best_model_state = model.state_dict()  # Save best model
    else:
        counter += 1  # Increase counter if no improvement

    # Stop training if patience is exceeded
    if counter >= patience:
        # print(f"Early stopping activated at epoch {i}")
        epochs = i
        break

    # step the scheduler after each epoch to update the learning rate
    # scheduler.step()

    if i < warmup_epochs:
        # Linearly increase LR
        warmup_lr = warmup_start_lr + (target_lr - warmup_start_lr) * (i / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    else:
        scheduler.step()  # use normal scheduler after warmup

# load the best model before early stopping
model.load_state_dict(best_model_state)

########################################################################
# LOSS CURVES

### LOSS CURVES ###
# epochs = 50
# plot training and validation loss over epochs

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 6), 
    gridspec_kw={'height_ratios': [3, 1]},  # 75% top, 25% bottom
    sharex=True  # share the x-axis (epochs)
)

# Plot training and validation loss
ax1.plot(range(len(losses)), losses, label='Training Loss')
ax1.plot(range(len(losses_val)), losses_val, label='Validation Loss')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss (with regularization)')
ax1.legend()
ax1.grid(True)

# Plot residuals (training loss / validation loss)
residuals = [train / val if val != 0 else float('nan') for train, val in zip(losses, losses_val)]
ax2.plot(range(len(losses)), residuals, color='purple')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Residual')
ax2.set_title('Training / Validation Loss')
ax2.grid(True)

# Add text box with final residual value
final_residual = residuals[-1]
textstr = f'Final Residual: {final_residual:.5f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax2.text(0.95, 0.1, textstr, transform=ax2.transAxes, fontsize=10, bbox=props,
          ha='right', va='bottom')

plt.tight_layout()
fig.savefig(plot_folder + 'losses.png')
# plt.show()

# overfitting: low training loss but a high or increasing validation loss
# underfitting: both high training loss and high validation loss

### ROC CURVE AND AUC ###
train_fpr, train_tpr, _ = roc_curve(y_train_tensor.cpu().numpy(), y_train_calc.detach().cpu().numpy())
val_fpr, val_tpr, _ = roc_curve(y_val.cpu().numpy(), y_val_calc.detach().cpu().numpy())

# Plot ROC curve
plt.plot(train_fpr, train_tpr, label=f'Train AUC = {roc_auc_score(y_train_tensor.cpu().numpy(), y_train_calc.detach().cpu().numpy()):.2f}')
plt.plot(val_fpr, val_tpr, label=f'Val AUC = {roc_auc_score(y_val.cpu().numpy(), y_val_calc.detach().cpu().numpy()):.2f}')
plt.plot([0, 1], [0, 1], 'k--', label="Random")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig(plot_folder + 'roc_curve.png')
# plt.show()

### PRECISION RECALL CURVE ###
from sklearn.metrics import precision_recall_curve, auc
# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_val.cpu().numpy(), y_val_calc.detach().cpu().numpy())
pr_auc = auc(recall, precision)

# Find the index of the closest threshold to best_threshold (0.14)
best_threshold = 0.14
best_idx = (thresholds >= best_threshold).argmax()
best_precision = precision[best_idx]
best_recall = recall[best_idx]

# print(f"Precision at threshold {best_threshold}: {best_precision:.4f}")
# print(f"Recall at threshold {best_threshold}: {best_recall:.4f}")

# Plot PR Curve
plt.figure()
plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
# plt.scatter(best_recall, best_precision, color='red', label=f'Best Threshold ({best_threshold:.2f})', zorder=3)  # Highlight best threshold
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig(plot_folder + 'pr_curve.png')
# plt.show()

### CALIBRATION CURVE ###
# true labels and predicted probabilities
true = y_val.cpu().numpy()
pred_probs = torch.sigmoid(y_val_calc).detach().cpu().numpy()

# get calibration curve (10 bins by default)
prob_true, prob_pred = calibration_curve(true, pred_probs, n_bins=30)

# plot
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequency')
plt.title('Calibration Curve')
plt.legend()
plt.savefig(plot_folder + 'calibration_curve.png')
# plt.show()

########################################################################
# MODEL EVALUATION

# make predictions on 2023-24 data
model.eval()
with torch.no_grad(): # basically turn off back propagation
    # y_pred = model(torch.tensor(X_predict_postprocess_df.to_numpy(),dtype=torch.float32))
    # y_pred = torch.sigmoid(y_pred)
    y_pred_logits = model(torch.tensor(X_predict_postprocess_df.to_numpy(),dtype=torch.float32))
    T = 1.0 # temperature to scale sigmoid
    y_pred = torch.sigmoid(y_pred_logits/T)

number_of_bins = int(np.sqrt(np.shape(data)[0])/8)

plt.figure(dpi = 100)
plt.hist(y_pred.numpy(), bins = number_of_bins)
plt.xlabel('Probability of goal')
plt.ylabel('Number of shots')
plt.savefig(plot_folder + 'xgoal_hist.png')
# plt.show()
########################################################################
# LOOK AT RESULTS

'''
add the calculated xgoal to the dataframe and compare
'''

pd.set_option("display.max_rows", None)

df = data_predict

pred_xgoal = y_pred

df['pred_xgoal'] = pred_xgoal

# for col in df.keys():print('{:<50}  {:>50}'.format(col , df[col][0]))

grouped_pred = df.groupby('shooterName').agg({'xGoal': 'sum', 'goal': 'sum', 'pred_xgoal': 'sum'})
grouped_pred['goal_residual'] = grouped_pred['goal'] - grouped_pred['xGoal']
grouped_pred['pred_goal_residual'] = grouped_pred['goal'] - grouped_pred['pred_xgoal'] # add new col that is total goals - xgoals
grouped_pred['residual_difference'] = grouped_pred['pred_goal_residual'] - grouped_pred['goal_residual']    
    
# print biggest discrepencies between goals and xgoals
sorted_grouped_pred = grouped_pred.sort_values(by='pred_goal_residual', ascending=True)

# print out the top 30 shooterName values, their cumulative xGoal, and sum of 'goal'
print('shooterName             AG: cumulative xGoal           Goals    Goals - xGoals')
print('--------------------------------------------------------------------------------')
for index, row in sorted_grouped_pred.head(20).iterrows():
    print('{:<25}  {:>15}  {:>15}  {:>15}'.format(index, round(row['pred_xgoal'],3), int(row['goal']), round(row['pred_goal_residual'],3)))
    
print()
    
print('shooterName             AG: cumulative xGoal           Goals    Goals - xGoals')
print('--------------------------------------------------------------------------------')
for index, row in sorted_grouped_pred.tail(20)[::-1].iterrows():
    print('{:<25}  {:>15}  {:>15}  {:>15}'.format(index, round(row['pred_xgoal'],3), int(row['goal']), round(row['pred_goal_residual'],3)))

# filter shooters with at least 20 goals
min_goal_for_chart = 25
filtered_shooters = sorted_grouped_pred[sorted_grouped_pred['goal'] >= min_goal_for_chart]
# sort by absolute residual values and take the top 20 closest to 0
closest_to_zero = filtered_shooters.iloc[filtered_shooters['pred_goal_residual'].abs().argsort()[:30]]
print(f'Most accurate predicted players (min {min_goal_for_chart} goals):')
print('shooterName             AG: cumulative xGoal           Goals    Goals - xGoals')
print('--------------------------------------------------------------------------------')

for index, row in closest_to_zero.iterrows():
    print('{:<25}  {:>15}  {:>15}  {:>15}'.format(index, round(row['pred_xgoal'],3), int(row['goal']), round(row['pred_goal_residual'],3)))

# print best performers according to model
sorted_grouped_pred_actual_goals = grouped_pred.sort_values(by='pred_xgoal', ascending=False)

# print out the top 30 shooterName values, their cumulative xGoal, and sum of 'goal'
print('shooterName             AG: cumulative xGoal           Goals    Goals - xGoals')
print('--------------------------------------------------------------------------------')
for index, row in sorted_grouped_pred_actual_goals.head(30).iterrows():
    print('{:<25}  {:>15}  {:>15}  {:>15}'.format(index, round(row['pred_xgoal'],3), int(row['goal']), round(row['pred_goal_residual'],3)))


# Extract x and y data
x = grouped_pred['goal'].values
y = grouped_pred['pred_xgoal'].values

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Generate regression line
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = slope * x_vals + intercept

# Compute standard error of predictions
# Residual standard error (RSE)
residuals = y - (slope * x + intercept)
rse = np.sqrt(np.sum(residuals**2) / (len(x) - 2))

# Mean x value and sample size
x_mean = np.mean(x)
n = len(x)

# Standard error of the regression line at each x
se_line = rse * np.sqrt(1/n + (x_vals - x_mean)**2 / np.sum((x - x_mean)**2))

# 95% confidence interval (approx. 1.96 * SE for normal distribution)
ci = 1.96 * se_line
lower = y_vals - ci
upper = y_vals + ci

# Plot Actual vs. Predicted
plt.figure(figsize=(10, 5), dpi=400)
sns.scatterplot(x=x, y=y, color='k', alpha=0.7, label="Data")

# Line of perfect predictions
sns.lineplot(x=x, y=x, color='cyan', linestyle='--', label="Ideal Predictions")

# Regression line with confidence interval
sns.lineplot(x=x_vals, y=y_vals, color='blue', label="Regression Line")
plt.fill_between(x_vals, lower, upper, color='blue', alpha=0.2, label="95% CI")

# Regression equation annotation
eq_text = f"y = {slope:.2f}x + {intercept:.2f}\n$R^2$ = {r_value**2:.2f}"
plt.text(0.05, 0.85, eq_text, transform=plt.gca().transAxes,
         fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.xlabel("Actual Goals")
plt.ylabel("Predicted Goals")
# plt.title("Actual vs. Predicted Goals: Unseen Data"
plt.title(f"Actual vs. Predicted Goals: Unseen Data, with T = {T}")
plt.legend()
plt.savefig(plot_folder + 'regression_line_pred_vs_real.png')
# plt.show()

# plot distrubtions of xgoals and goals

bins__ = 50

plt.figure(dpi = 100)
plt.title('Compare distributions of goals vs xgoals')
plt.hist(grouped_pred['goal'], bins = bins__,histtype = 'step',color = 'blue',label = 'goals')
plt.hist(grouped_pred['pred_xgoal'], bins = bins__,histtype = 'step',color = 'red',label = 'xgoals')
plt.xlabel('Goals scored')
plt.legend()
plt.savefig(plot_folder + 'compare_distributions.png')
# plt.show()

print(describe(grouped_pred['goal']))
print(describe(grouped_pred['pred_xgoal']))

# plt.figure(dpi = 100)
# plt.hist(grouped_pred['residual_difference'], bins = bins__,histtype = 'step',color = 'blue',label = 'residual between models')
# plt.xlabel('AG - MP')
# plt.legend()
# plt.show()

# print(describe(grouped_pred['residual_difference']))

plt.figure(dpi = 100)
plt.title('Compare AG model vs MP model')
plt.hist(grouped_pred['goal_residual'], bins = bins__,histtype = 'step',color = 'blue',label = 'MP')
plt.hist(grouped_pred['pred_goal_residual'], bins = bins__,histtype = 'step',color = 'red',label = 'AG')
plt.xlabel('goals - xgoals')
plt.legend()
plt.savefig(plot_folder + 'ag_vs_mp.png')
# plt.show()

print(describe(grouped_pred['goal_residual']))
print(describe(grouped_pred['pred_goal_residual']))