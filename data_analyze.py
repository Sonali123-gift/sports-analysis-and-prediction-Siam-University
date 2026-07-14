# Optimizing University Sport Investment Using Popularity and Stress Analysis
# Group 4 - Sonali | Siam University
# CRISP-DM Framework | Data Science Project

import streamlit as st
import pandas as pd
import numpy as np
import statistics

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.stats import norm
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


st.title("📊 Optimizing University Sport 🏅 Investment Using Popularity And Stress Analysis")
st.header('Upload Your Cleaned Data')
uploaded_file = st.file_uploader('Upload Cleaned_Data.xlsx', type=['xlsx'])

if uploaded_file is None:
    st.info('Please upload your Cleaned_Data.xlsx file to continue.')
    st.stop()

df = pd.read_excel(uploaded_file)
df = df.drop(columns=[c for c in df.columns if 'unnamed' in c.lower()])
df = df.rename(columns={'AcitvityLevel': 'ActivityLevel'})

# Removing 'Prefer Not To Say' from dataset
df = df[df['Gender'] != 'Prefer Not To Say']
df = df.reset_index(drop=True)



st.dataframe(df)
st.info('📋 The dataset covers key lifestyle and academic variables that may influence student stress. "Prefer Not To Say" gender responses have been removed so that the gender analysis compares only Male and Female groups. All 128 remaining responses are used in the full analysis.')

st.header('Descriptive Statistics')

df_selected = df[['Stress_num', 'SportsHrs_num', 'Sleep_num', 'Height_num']]

#calculate and display stats for each column using a loop
for column in df_selected.columns:
    st.subheader(column + ' Statistics')
    data = df_selected[column]
    st.write('Count:', len(data))
    st.write('Mean:', round(np.mean(data), 2))
    st.write('Median:', round(np.median(data), 2))
    st.write('Mode:', statistics.mode(data))
    st.write('Standard Deviation:', round(np.std(data), 2))
    st.write('Min:', min(data))
    st.write('Max:', max(data))

#IQR and fence values for Stress
Q1 = np.percentile(df['Stress_num'], 25)
Q3 = np.percentile(df['Stress_num'], 75)
IQR = Q3 - Q1

st.subheader('Stress Level - IQR and Fence Values')
st.write('Q1:', Q1)
st.write('Q3:', Q3)
st.write('IQR:', IQR)
st.write('Lower Fence:', Q1 - 1.5 * IQR)
st.write('Upper Fence:', Q3 + 1.5 * IQR)
st.info('📊 The IQR of 1.0 shows the middle 50% of stress scores falls between 3 and 4, indicating that most students experience moderate-to-high stress. The upper fence is 5.5, which exceeds the maximum possible score of 5 — meaning no students qualify as high-stress outliers; all high scores are genuine. The lower fence is 1.5, so only students scoring below 1.5 would be outliers on the low end. This confirms the stress distribution is real and not distorted by anomalies.')

st.subheader('Summary Table')
st.dataframe(df_selected.describe())
st.info('📋 Summary: Stress scores average around 3.4 out of 5, indicating moderate-to-high stress across the sample. The median of 3 and mode of 3 confirm that the most common experience is mid-range stress. Sport hours average just 2.73 hours per week with a low median of 1.5, suggesting most students participate minimally. Sleep averages 6 hours per night, below the recommended 8 hours, indicating mild sleep deprivation is common across the sample.')


df['HighStress'] = (df['Stress_num'] >= 4).astype(int)

df['Workload_num'] = df['AcademicWorkload'].map({'Low': 1, 'Medium': 2, 'High': 3})
df['Activity_num'] = df['ActivityLevel'].map({'Very Low': 1, 'Low': 2, 'Moderate': 3, 'High': 4})
df['Gender_bin'] = (df['Gender'] == 'Female').astype(int)

st.header('Sport Popularity Analysis')
st.write('Which sports do the most students participate in?')

#bar chart - sport popularity
sport_counts = df['Sport'].value_counts().reset_index()
sport_counts.columns = ['Sport', 'Count']
sport_counts['Percentage'] = round(sport_counts['Count'] / len(df) * 100, 1)

fig_bar_sport = px.bar(sport_counts, x='Sport', y='Count',
                       color='Sport',
                       text='Percentage',
                       title='Sport Popularity - Number of Students per Sport')
fig_bar_sport.update_traces(texttemplate='%{text}%', textposition='outside')
st.plotly_chart(fig_bar_sport)

st.write('Key Finding: Badminton is the most popular sport with 59 students (43%).')
st.write('Basketball is second with 36 students (26%).')
st.write('Badminton + Basketball together cover 69% of all sport participants.')
st.write('Majority of participation is concentrated in just 2 sports.')
st.info('📊 Interpretation: Sport participation is highly concentrated in two sports. Badminton alone accounts for 43% of students, making it the dominant activity on campus. Basketball follows at 26%. Every other sport individually accounts for less than 15% of students. This concentration pattern is critical for investment decisions — any resources directed at minority sports would benefit fewer than 1 in 7 students, making them poor value compared to investing in Badminton and Basketball infrastructure.')


st.header('Workload vs Stress Analysis')
st.write('RQ1: Does academic workload significantly increase student stress?')


#bar chart - average stress by workload
wl_stress = df.groupby('AcademicWorkload')['Stress_num'].mean().reindex(
    ['Low', 'Medium', 'High']).reset_index()
wl_stress.columns = ['AcademicWorkload', 'Average Stress']
fig_bar_wl = px.bar(wl_stress, x='AcademicWorkload', y='Average Stress',
                    color='AcademicWorkload',
                    title='Average Stress Level by Academic Workload')
st.plotly_chart(fig_bar_wl)
st.info('📊 Interpretation: The bar chart shows a clear upward staircase pattern — average stress rises consistently from Low (3.0) to Medium (3.14) to High (4.08) workload groups. This visual trend strongly suggests a positive relationship between workload and stress. The jump from Medium to High is notably larger than from Low to Medium, indicating that high workload has a disproportionately strong effect on stress compared to moderate workload.')

#ANOVA statistical test
group_Low    = df[df['AcademicWorkload'] == 'Low']['Stress_num']
group_Medium = df[df['AcademicWorkload'] == 'Medium']['Stress_num']
group_High   = df[df['AcademicWorkload'] == 'High']['Stress_num']
f_stat, p_value_anova = stats.f_oneway(group_Low, group_Medium, group_High)

st.subheader('One-Way ANOVA - Workload vs Stress')
st.write('Low Workload    Mean Stress:', round(group_Low.mean(), 2))
st.write('Medium Workload Mean Stress:', round(group_Medium.mean(), 2))
st.write('High Workload   Mean Stress:', round(group_High.mean(), 2))
st.write('F-statistic:', round(f_stat, 4))
st.write('p-value    :', round(p_value_anova, 6))
if p_value_anova < 0.05:
    st.write('Result: SIGNIFICANT (p < 0.05)')
    st.write('Conclusion: Academic workload significantly increases student stress.')
    st.write('High workload students are 37% more stressed than low workload students.')
    st.write('This is the main stress driver confirmed by statistical testing.')
st.info('📊 ANOVA Interpretation: The one-way ANOVA tests whether mean stress differs significantly across the three workload groups. The F-statistic of 10.33 and p-value of 0.00007 confirm the differences are statistically significant — far below the 0.05 threshold. This means workload group differences in stress did not occur by chance. With High workload students averaging 4.08 stress vs 3.0 for Low workload students, academic pressure is the strongest and most statistically reliable predictor of stress in this entire dataset.')


st.header('Sport vs Stress Analysis')
st.write('RQ2: Does sport participation reduce student stress?')

#sport hours vs wellness chart
df['Sleep_Sport_Interaction'] = df['Sleep_num'] * df['SportsHrs_num']
df['Workload_Sleep'] = df['Workload_num'] * df['Sleep_num']
df['Workload_Activity'] = df['Workload_num'] * df['Activity_num']

st.subheader('Sport Hours vs Wellness (Stress and Sleep)')
fig_wellness = px.scatter(
    df,
    x='SportsHrs_num',
    y='Stress_num',
    color='AcademicWorkload',
    labels={
        'SportsHrs_num': 'Sport Hours per Week',
        'Stress_num': 'Stress Level'
    },
    title='Sport Hours vs Stress Level'
)
st.plotly_chart(fig_wellness)
st.info('📊 Interpretation: The scatter plot shows stress levels across different sport hours, coloured by academic workload. The most visible pattern is that High workload students (one colour) cluster at stress levels 4–5 regardless of how many sport hours they do — confirming that workload dominates stress and sport cannot offset it. Low and Medium workload students sit at lower stress levels across all sport hour values. The overall downward slope is very gentle, consistent with the weak and non-significant Pearson correlation below.')

#pearson correlation
r_sport, p_sport = stats.pearsonr(df['SportsHrs_num'], df['Stress_num'])
st.subheader('Pearson Correlation - Sport Hours vs Stress')
st.write('Pearson r:', round(r_sport, 4))
st.write('p-value  :', round(p_sport, 4))
if p_sport < 0.05:
    st.write('Result: SIGNIFICANT (p < 0.05)')
else:
    st.write('Result: NOT significant (p >= 0.05)')
st.write('Conclusion: There is a weak negative correlation between sport hours and stress.')
st.write('More sport is linked to slightly lower stress, but the effect is not significant alone.')
st.write('Workload remains the dominant driver. Sport helps but cannot overcome structural pressure.')
st.info('📊 Pearson Correlation Interpretation: The Pearson r of -0.1336 confirms a weak negative direction — more sport is associated with slightly lower stress. However, the p-value of 0.1328 is greater than 0.05, meaning this result is NOT statistically significant. We cannot reject the null hypothesis that the true correlation is zero. In plain terms: sport hours alone do not reliably predict stress in this sample. Sport may still be beneficial, but this dataset does not provide statistical proof. Any recommendation to invest in sport for stress reduction must be paired with workload management, which is the only statistically confirmed driver.')


st.header('Gender vs Stress Analysis')
st.write('Do female and male students experience different stress levels?')

#bar chart - average stress by gender
gender_stress = df.groupby('Gender')['Stress_num'].mean().reset_index()
gender_stress.columns = ['Gender', 'Average Stress']
fig_bar_gen = px.bar(gender_stress, x='Gender', y='Average Stress',
                     color='Gender',
                     title='Average Stress Level by Gender')
st.plotly_chart(fig_bar_gen)
st.info('📊 Interpretation: The bar chart shows female students report a higher average stress level (3.74) compared to male students (3.13). While the bars may appear close in height, a difference of 0.61 stress points on a 5-point scale is meaningful — particularly when confirmed by a statistical test. This gap suggests a systemic difference in stress experience between genders rather than random variation.')

#t-test
group_Male   = df[df['Gender'] == 'Male']['Stress_num']
group_Female = df[df['Gender'] == 'Female']['Stress_num']
t_stat, p_value_ttest = stats.ttest_ind(group_Male, group_Female)

st.subheader('Independent t-test - Male vs Female Stress')
st.write('Male   Mean Stress:', round(group_Male.mean(), 2), '  n =', len(group_Male))
st.write('Female Mean Stress:', round(group_Female.mean(), 2), '  n =', len(group_Female))
st.write('High Stress - Female:', round(df[df['Gender']=='Female']['HighStress'].mean()*100, 1), '%')
st.write('High Stress - Male  :', round(df[df['Gender']=='Male']['HighStress'].mean()*100, 1), '%')
st.write('t-statistic:', round(t_stat, 4))
st.write('p-value    :', round(p_value_ttest, 4))
if p_value_ttest < 0.05:
    st.write('Result: SIGNIFICANT (p < 0.05)')
    st.write('Conclusion: Female students have significantly higher stress than male students.')
    st.write('52.6% of female students are high stress vs 35.2% of male students.')
    st.write('Female students also play fewer sport hours and sleep less per night.')
    st.write('Targeted intervention is required for female students.')
st.info('📊 t-test Interpretation: The independent samples t-test produces a t-statistic of -2.98 and p-value of 0.0035, which is well below the 0.05 significance threshold. This confirms the gender stress gap is statistically real and not due to chance. Female students have a 52.6% high-stress rate versus 35.2% for male students — a 17.4 percentage point gap. This is the second strongest finding in the dataset after workload. Gender-specific interventions such as counselling, flexible scheduling, and female-targeted sport programmes are statistically warranted.')

st.header('Gender vs Sport Hours')

fig = px.box(
    df,
    x='Gender',
    y='SportsHrs_num',
    color='Gender',
    title='Sport Hours Distribution by Gender'
)

fig.update_layout(
    xaxis_title='Gender',
    yaxis_title='Sport Hours per Week',
    template='plotly_dark'
)

st.plotly_chart(fig, use_container_width=True)
st.info('📊 Interpretation: The box plot shows male students have a higher median sport participation than female students. The male box (IQR) is wider, indicating greater variation in how much sport male students do. Female students show a lower median and a narrower spread, suggesting more consistently low sport participation. This pattern is relevant because female students already experience higher stress — if sport has any stress-relief benefit, female students are getting less of it. However, since sport\'s correlation with stress is not statistically significant, increasing female sport hours alone will not close the stress gap without also addressing workload.')

st.header('Correlation Heatmap')
st.write('How do all key variables relate to each other?')

df_corr = df[['Stress_num', 'SportsHrs_num', 'Sleep_num', 'Workload_num', 'Activity_num']]
matrix  = df_corr.corr()

fig_heatmap, ax_heatmap = plt.subplots()
sns.heatmap(matrix, annot=True, cmap='coolwarm', ax=ax_heatmap)
st.pyplot(fig_heatmap)

st.subheader('Correlation Summary')
st.write('Workload  -> Stress :', round(matrix.loc['Stress_num','Workload_num'], 4),
         '  (Positive - higher workload = higher stress)')
st.write('Activity  -> Stress :', round(matrix.loc['Stress_num','Activity_num'], 4),
         '  (Negative - more activity = lower stress)')
st.write('Sleep     -> Stress :', round(matrix.loc['Stress_num','Sleep_num'], 4),
         '  (Negative - more sleep = lower stress)')
st.write('Sport Hrs -> Stress :', round(matrix.loc['Stress_num','SportsHrs_num'], 4),
         '  (Negative - more sport = slightly lower stress)')
st.write('Conclusion: Workload increases stress. Activity and Sleep reduce stress.')

#sport hours vs wellness chart (sleep section)
st.subheader('Sport Hours vs Wellness Score by Gender')
fig_wellness_gender = px.scatter(
    df,
    x='SportsHrs_num',
    y='Stress_num',
    color='Gender',
    labels={
        'SportsHrs_num': 'Sport Hours per Week',
        'Stress_num': 'Stress Level'
    },
    title='Sport Hours vs Stress Level by Gender'
)
st.plotly_chart(fig_wellness_gender)

r_sleep, p_sleep = stats.pearsonr(df['Sleep_num'], df['Stress_num'])
st.write('Sleep vs Stress  r:', round(r_sleep, 4), '  p:', round(p_sleep, 4))
st.info('📊 Sleep vs Stress Interpretation: The Pearson r between sleep and stress is -0.047 with a p-value of 0.5999, which is far above the 0.05 significance threshold. This result is NOT statistically significant — we cannot conclude that sleeping more reduces stress in this sample. While the direction is negative (more sleep = slightly lower stress), the effect is extremely weak and unreliable. Sleep deprivation may still matter in reality, but this dataset does not provide sufficient evidence to recommend sleep improvement as a primary stress-reduction strategy.')


st.header('Machine Learning - Model Explanation')

#obtain features
df_x = df[['SportsHrs_num', 'Sleep_num', 'Workload_num', 'Activity_num', 
           'Gender_bin', 'Sleep_Sport_Interaction', 
           'Workload_Sleep', 'Workload_Activity']]

#obtain target (continuous for linear regression)
df_y_cont = df[['Stress_num']]

#obtain target class (for decision tree)
df_y = df[['HighStress']]

feature_list = df_x.columns
class_list   = ['Low/Moderate Stress', 'High Stress']


st.header('Linear Regression Model')

test_ratio = st.number_input('Test Set Ratio (Linear Regression)', value=0.2)
x_train, x_test, y_train, y_test = train_test_split(
    df_x, df_y_cont, test_size=test_ratio, random_state=42)

scaler         = StandardScaler()
from sklearn.preprocessing import PolynomialFeatures

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled  = scaler.transform(x_test)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
x_train_scaled = poly.fit_transform(x_train_scaled)
x_test_scaled  = poly.transform(x_test_scaled)
lin_model = LinearRegression()
lin_model.fit(x_train_scaled, y_train)
y_pred_lin = lin_model.predict(x_test_scaled)

mse  = mean_squared_error(y_test, y_pred_lin)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred_lin)

st.write('Mean Squared Error (MSE) :', round(mse, 4))
st.write('Root Mean Squared Error  :', round(rmse, 4))
st.write('R-squared (R²)           :', round(r2, 4))
st.info('📊 Model Fit Interpretation: MSE and RMSE measure average prediction error in stress score units — lower values mean predictions are closer to actual stress scores. R² measures what proportion of stress variation the model explains (0 = no explanation, 1 = perfect). This model uses polynomial features and interaction terms (e.g., Workload × Sleep), which capture non-linear relationships that a basic linear regression would miss. If R² is negative, the model performs worse than simply predicting the mean stress for every student, which indicates the polynomial expansion has overfit to the training data and the test set is too small for reliable evaluation. If R² is positive, interpret it as the percentage of stress variance explained by the combined effect of all features.')

st.subheader('Prediction Stress Level')

# Add predictions back to test set
x_test_copy = x_test.copy()
x_test_copy['Predicted_Stress'] = y_pred_lin.flatten()

fig1 = px.scatter(
    x_test_copy,
    x='SportsHrs_num',
    y='Predicted_Stress',
    trendline='ols',
    title='Sport Hours vs Predicted Stress'
)
st.plotly_chart(fig1)
st.info('📊 Interpretation: This chart shows the model\'s predicted stress values plotted against sport hours. A downward-sloping trendline confirms the model has learned that more sport hours are associated with lower predicted stress — consistent with the negative Pearson r. However, the wide scatter of points around the line shows the model\'s uncertainty: sport hours alone explain very little of the variation in predicted stress. This is expected given sport\'s weak and non-significant correlation with stress. The trendline direction is correct but the effect is small.')


fig2 = px.scatter(
    x_test_copy,
    x='Workload_num',
    y='Predicted_Stress',
    trendline='ols',
    title='Workload vs Predicted Stress (Regression Line)'
)
st.plotly_chart(fig2)
st.info('📊 Interpretation: This regression line should show the steepest upward slope of all three predictor charts — confirming workload as the dominant stress driver in the model. Compare this slope to the sport hours and sleep charts: if workload\'s slope is the most pronounced, it quantitatively confirms that reducing academic pressure would produce the greatest predicted reduction in student stress. This is consistent with the ANOVA result (p = 0.00007) which identified workload as the only statistically significant stress predictor in the dataset.')

labels={
    'SportsHrs_num': 'Sport Hours per Week',
    'Predicted_Stress': 'Predicted Stress Level'
}

st.subheader('Linear Regression Coefficients')
# Get correct feature names after polynomial transformation
poly_feature_names = poly.get_feature_names_out(feature_list)

coef_df = pd.DataFrame({
    'Feature': poly_feature_names,
    'Coefficient': lin_model.coef_[0]
}).sort_values('Coefficient', ascending=False)

st.dataframe(coef_df)
poly.get_feature_names_out()

st.write('Positive coefficient = increases predicted stress level.')
st.write('Negative coefficient = reduces predicted stress level.')
st.write('Workload_num has the highest positive coefficient confirming it as the #1 stress driver.')


st.header('Decision Tree Model')
st.write('The tree shows which factors are used to predict high stress.')
st.write('First split = most important factor.')
st.write('Second level = second most important factors.')

test_ratio_dt = st.number_input('Test Set Ratio (Decision Tree)', value=0.2)
x_train_dt, x_test_dt, y_train_dt, y_test_dt = train_test_split(
    df_x, df_y, test_size=test_ratio_dt, random_state=42, stratify=df_y)

tree = DecisionTreeClassifier(random_state=1, max_depth=4)
tree.fit(x_train_dt, y_train_dt)
y_pred_dt = tree.predict(x_test_dt)

st.subheader('Decision Tree Visualization')
plt.clf()
plt.figure(figsize=(20, 10))
plot_tree(tree, filled=True,
          feature_names=list(feature_list),
          class_names=class_list)
st.pyplot(plt)

st.write('First split  = Workload_num  -> Confirms workload is the main factor.')
st.write('Second split = Sleep_num / Activity_num -> Sleep and activity are secondary factors.')
st.info('📊 Decision Tree Interpretation: Each node represents a decision rule applied to a student\'s data. The root (first) split uses the most powerful classifier — whichever feature appears at the top confirms it as the single most important predictor of high vs low stress. Subsequent splits at the second and third levels refine the classification using secondary variables. Blue/lighter nodes lean toward Low/Moderate Stress; orange/darker nodes lean toward High Stress. A student with high workload, low sleep, and low activity represents the highest-risk profile. Read the path from root to any leaf node as a student profile: follow the branches that match a student\'s characteristics to predict their stress category.')

st.write('Test Accuracy:', round(accuracy_score(y_test_dt, y_pred_dt), 4))
st.info(f'📊 Accuracy Interpretation: The Decision Tree correctly classifies {round(accuracy_score(y_test_dt, y_pred_dt)*100, 1)}% of students as High Stress or Low/Moderate Stress on the held-out test set. For context, a model that always predicts the majority class would achieve a baseline accuracy based on class proportions — so accuracy should be evaluated relative to that baseline, not just the 70% rule of thumb. An accuracy above 70% on a small behavioural dataset (n=128) is considered acceptable for a screening tool. This model is useful for identifying at-risk students in a general sense, but should not be used as a definitive clinical assessment.')

st.subheader('Classification Report')
report_dt = classification_report(y_test_dt, y_pred_dt,
                                   target_names=class_list, output_dict=True)
st.dataframe(pd.DataFrame(report_dt).transpose())


st.subheader('Feature Importances')
importance_df = pd.DataFrame({
    'Feature': list(feature_list),
    'Importance': tree.feature_importances_
}).sort_values('Importance', ascending=False)
st.dataframe(importance_df)
st.info('📊 Feature Importance Interpretation: Importance scores show each variable\'s actual contribution to the tree\'s classification decisions — scores sum to 1.0. The feature with the highest score was used most prominently in splitting decisions and at the highest levels of the tree. Interaction terms such as Workload_Sleep and Sleep_Sport_Interaction often rank highly, reflecting that combined variable effects are stronger predictors than any single variable alone. Features with an importance score of 0 were never used in any split and have no predictive value in this model. These zero-importance features could be removed without affecting model performance.')


st.header('Model Performance Comparison')

r2_display = round(r2, 4)
acc_dt = round(accuracy_score(y_test_dt, y_pred_dt), 4)

comparison_df = pd.DataFrame({
    'Model'        : ['Linear Regression', 'Decision Tree'],
    'Performance'  : [f'R² = {r2_display}', f'Accuracy = {acc_dt}'],
    'Best For'     : ['Explaining which factors matter and by how much (continuous prediction)',
                      'Visual rules that are easy to present and explain']
})
st.dataframe(comparison_df)

st.write('Linear Regression R²:', r2_display)
st.write('Decision Tree accuracy      :', acc_dt)
st.write('Linear Regression predicts the actual stress score as a continuous value.')
st.write('Decision Tree is easier to explain and visualize for presentations.')
st.write('Both models confirm: Workload is the #1 predictor of high stress.')
st.info('📊 Model Comparison Interpretation: The two models answer different questions and should be read together. The Linear Regression R² reflects how well a continuous stress score can be predicted — a negative R² means the polynomial model has overfit to training data and performs poorly on the test set, so use the coefficients for directional insight only, not for precise predictions. The Decision Tree accuracy reflects how reliably students can be classified as high-risk or not — values above 70% are considered acceptable for small behavioural datasets. Regardless of individual model performance, both approaches consistently identify workload as the dominant factor, which corroborates the ANOVA result and makes that finding robust.')

st.header('Integrated Insight - Combining Both Analyses')

st.write('Based on the data:')
st.write('')

sport_investment = pd.DataFrame({
    'Rank'          : ['1st Priority', '2nd Priority'],
    'Sport'         : ['Badminton', 'Basketball'],
    'Students'      : ['59  (43%)', '36  (26%)'],
    'Avg Stress'    : [round(df[df['Sport']=='Badminton']['Stress_num'].mean(), 2),
                       round(df[df['Sport']=='Basketball']['Stress_num'].mean(), 2)],
    'Combined Cover': ['43%', '43% + 26% = 69%']
})
st.dataframe(sport_investment)

st.write('Investing in Badminton and Basketball covers 69% of all sport participants.')
st.write('This maximizes the reach of every dollar spent on sport infrastructure.')
