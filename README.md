# mushroomclassification

#### Problem Statement
To classify mushroom samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family as edible or poisonous.

#### Purpose of the solving this problem

Creating a mushroom classification model to detect if a mushroom is edible or not serves several important purposes:

##### 1) Safety and Health

###### Prevent Poisoning: 
Mushrooms can be highly toxic and even deadly if consumed. A classification model can help identify edible versus poisonous mushrooms, reducing the risk of poisoning and potentially saving lives.

###### Public Health: 
Provides a tool for both amateur and professional foragers to ensure that the mushrooms they collect are safe to eat, contributing to public health safety.

##### 2. Educational and Informational

###### Awareness: 
Educates people about different mushroom species, their edibility, and their characteristics.

###### Training Tool: 
Can be used in educational settings to teach about mushroom classification and mycology.

##### 3. Commercial Applications

###### Food Industry: 
Assists in quality control and safety checks in the commercial mushroom farming and distribution industry.

###### Consumer Products: 
Helps in developing consumer-friendly applications for identifying mushrooms in the wild.

##### 4. Research and Conservation

###### Species Identification: 
Contributes to research on mushroom species, their distribution, and conservation needs.

###### Biodiversity Studies: 
Supports studies related to biodiversity by accurately identifying and classifying different mushroom species.

##### 5. Technology Integration

###### Mobile Apps: 
Powers mobile applications that allow users to take photos of mushrooms and receive instant classification, making the tool accessible to a broader audience.

###### Automation: 
Integrates with automated systems for foraging or harvesting mushrooms in controlled environments.

##### 6. Economic Impact
###### Market Value: 
Enhances the market value of edible mushrooms by ensuring they are correctly identified and classified.

###### Reducing Waste: 
Helps in reducing waste by preventing the harvest of inedible or toxic mushrooms.

##### Example Use Case
For example, in a mobile application designed for mushroom enthusiasts or foragers, a classification model could allow users to take a photo of a mushroom they encounter and instantly determine if it’s edible. This would empower users to make safer choices in the wild and promote the responsible enjoyment of mushrooms.

Overall, the primary goal of such a model is to enhance safety, support education, and facilitate accurate identification of mushroom species, thereby benefiting various sectors and individuals.

#### Dataset Information

The dataset has been downloaded from UC Irvine Machine Learning Repository.

https://archive.ics.uci.edu/dataset/73/mushroom

This data set includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms in the Agaricus and Lepiota Family (pp. 500-525).  Each species is identified as definitely edible, definitely poisonous, or of unknown edibility and not recommended.  This latter class was combined with the poisonous one.  The Guide clearly states that there is no simple rule for determining the edibility of a mushroom; no rule like ''leaflets three, let it be'' for Poisonous Oak and Ivy.

#### Instructions on how to run your code and reproduce the results

Step 1 : Go to the Anaconda website and download the installer for your operating system (Windows, macOS, or Linux) & launch it.
Step 2: Launch Jupyter Notebook Interface from Anaconda Navigator
Step 3: Create the Project folder with directories for data, notebooks, models and visuals.
Step 4: Create a new notebook in the notebooks folder inside the Project Folder
Step 5 : Install the UCI ML Repository to import the mushroom dataset
Step 6 : Import the libraries listed in the requirements.txt file
Step 7 : Load the dataset 
Step 8 : Understand how the data is structured, what it contains
Step 9 : Save the raw dataframes and it's attributes in their respective files in the raw data folder
Step 10 : Explore the datasets using head(), tail() and info() methods
Step 11 : Identify the target column and explore it's unique values
Step 12 : Identify columns which have only 1 value
Step 13 : Perform Data Preprocessing to merge the datasets 
Step 14 : Check the distribution of the target variable to see if the datset is balanced
Step 15 : Perform Feature Engineering to transform the target variable into a binary variable
Step 16 : Define a folder variable containing the path to the visuals folder
Step 17 : Perform Extensive Exploratory Data Analysis to check the distribution of all variables with respect to the target variable, explain the insights and save all the visuals to the visuals folder using plt.savefig()
Step 18 : Check missing values in the dataset
Step 19 : Impute the missing values using mode()
Step 20 : Look at the stats of the target variable if required using describe()
Step 21 : Create a copy of the dataframe before Encoding and Scaling for comparison purposes.
Step 22 : Segregate and Extract the numerical, categorical features 
Step 23 : Encode the categorical features into numeric values and check the dataset
Step 24 : Update the numerical, categorical features and separate the binary features from numerical ones
Step 25 : Scale the numeric features and check the dataset
Step 26 : Drop the columns which have just one value
Step 27 : Obtain a correlation matrix of the encoded-scaled dataset
Step 28 : Plot a Heatmap of the correlation matrix to check highly correalted features, save the plot to the visuals folder
Step 29 : Drop one of the highly corelated features from every highly-corelated pair, replot the heatmap and save it to the visuals folder
Step 30 : Extract the independent and dependent varaibles from the dataset into x & y respectively
Step 31 : Perform Train-Test-Split on the dataset to split it into training and testing sets.
Step 32 : Check the shape of the splits and also take a look at their data
Step 33 : Obtain an instance of the LogisticRegression model, fit it on the training data and check it's coefficeints
Step 34 : Run the model on the test data and obtain the predictions
Step 36 : Evaluate the Accuracy of the model using the accuracy()
Step 37 : Plot the confusion matrix of the model and store it to the visuals folder
Step 38 : Plot the ROC Curve of the model and store it to the visuals folder
Step 39 : Print the Classification Report and check the F1 Score of the model
Step 40 : Print and check the Log Loss of the model
Step 41 : Apply K-Fold Cross Validation to perform Hyperparameter tuning to find the best value of C (C = 1/Lambda) before performing Regularization
Step 42 : Print the best value of C
Step 43 : Plot the Cross Validation Accuracy for different values of C and store the visual to the visuals folder
Step 44 : Create an instance of the LogisticRegression model for Ridge Regularization using parameters for the best value of C, penalty as "l2" and solver as "liblinear"
Step 45 : Fit it on the training data and check it's coefficeints
Step 46 : Run the model on the test data, obtain the predictions and print it's accuracy
Step 47 : Create an instance of the LogisticRegression model for Lasso Regularization using parameters for the best value of C, penalty as "l1" and solver as "liblinear"
Step 48 : Repeat steps 45 & 46 for the new model
Step 49 : Create an instance of the LogisticRegression model for ElasticNet Regularization using parameters for the best value of C, penalty as "elasticnet", solver as "saga" and l1_ratio as 0.5
Step 50 : Repeat steps 45 & 46 for the new model
Step 51 : Create a folder variable to store the path of the models folder to store all models
Step 52 : Save all models in the models folder using joblib.dump()
Step 53 : Create a score-card data frame and define a function to update it with the metrics of all models
Step 54 : Call the update score card method created above for all 4 models created
Step 55 : Compare the results of the 4 models using the score-card to find the best performing model
Step 56 : Save and run the whole notebook
Step 57 : Check if the models have been saved to the models folder
Step 58 : Check if the visuals habe been saved to the visulas folder
Step 59 : Close the Jupyer Notebook Interface and the Annaconda Navigator.

#### Explanations of code and models

#### A) Fetching and Understanding the Mushroom dataset from the UCI Repository

##### I) Fetching and Understanding the Mushroom dataset from the UCI Repository

The comments mentioned throughout the notebook already explain the code and the models. However for the purpose of detailed understanding here is the detailed explanation.

from ucimlrepo import fetch_ucirepo
This line imports the fetch_ucirepo function from the ucimlrepo library, which is used to fetch datasets from the UCI Machine Learning Repository.

mushroom = fetch_ucirepo(id=73)
This line fetches the dataset with ID 73 from the UCI Machine Learning Repository and stores it in the variable mushroom. The dataset being fetched here is related to mushrooms (often used in classification tasks).

features = mushroom.data.features
targets = mushroom.data.targets
features contains the feature data of the dataset, which are the input variables used for machine learning.
targets contains the target data (or labels) of the dataset, which are the values you aim to predict.

print(mushroom.metadata)
This line prints metadata about the dataset. Metadata usually includes information such as the dataset’s description, the source, and any relevant details about how the data was collected or processed.

print(mushroom.variables)
This line prints information about the variables in the dataset. This can include details about each feature, such as its name, type, and possible values.

attributes = mushroom.variables
This line stores the variable information in the attributes variable for later use.

##### II) Specify File Paths and Save DataFrames:

For Features Data:

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/mushroomclassification/data/rawdata", "features_data.csv")
features.to_csv(file_path, index=False)
This specifies the file path where you want to save the features DataFrame as a CSV file and saves it using to_csv(). index=False ensures that the DataFrame index is not included in the saved file.

For Targets Data:

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/mushroomclassification/data/rawdata", "targets_data.csv")
targets.to_csv(file_path, index=False)
Similarly, this specifies the path for saving the targets DataFrame and writes it to a CSV file.

For Attributes Data:

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/mushroomclassification/data/rawdata", "README.csv")
attributes.to_csv(file_path, index=False)
This saves the attributes DataFrame, which contains variable information, to a CSV file named README.csv.

##### III) View Data:

First 5 Rows of Features:

features.head()
This displays the first 5 rows of the features DataFrame to give a quick overview of the data.

Unique Values in 'veil-type' Column:

features['veil-type'].unique()
This checks the unique values in the veil-type column to determine if there is only one unique value, which could indicate it’s not useful for analysis.

First 5 Rows of Targets:
targets.head()
This displays the first 5 rows of the targets DataFrame to understand its structure and content.

Unique Values in Target Variable:
targets['poisonous'].unique()
This checks the unique values in the poisonous target variable to understand the classes or labels present.

##### IV) Extract Comprehensive Information:
features.info()
This provides a concise summary of the features DataFrame, including the number of entries, column names, data types, and non-null counts.


#### B) Data Preprocessing and Feature Engineering

##### 1) Merging Features and Targets
df = features.join(targets)

Objective: Combine the features DataFrame with the targets DataFrame into a single DataFrame df.

Method: join() is used to merge the two DataFrames based on their indices. This assumes that the indices of features and targets align correctly.

##### 2) Checking Value Distribution in Target Variable

df['poisonous'].value_counts()

Objective: Check the distribution of unique values in the poisonous column of df.

Purpose: To confirm if the dataset is balanced with respect to the target variable, which is important for model performance. value_counts() returns a Series with counts of unique values in the poisonous column.

##### 3) Transforming Target Variable

df['poisonous'] = df['poisonous'].apply(lambda x: 0 if x == 'e' else 1)

Objective: Convert the poisonous column into a binary variable.

Method: apply() is used with a lambda function to transform the target variable:
If the value is 'e' (edible), it is converted to 0.
If the value is not 'e' (poisonous), it is converted to 1.

Result: This transformation makes the target variable suitable for binary classification tasks.

#### C) Exploratory Data Analysis

##### 1) Define Folder for Visualizations:

visuals_folder = "C:/Users/nikde/Documents/UpGrad/mushroomclassification/visuals"
Defines the path where the visualizations will be saved.

##### 2) Generate and Save Visualizations:

plt.figure(figsize=(14,2))
sns.histplot(data=df, x='habitat', hue='poisonous', palette={0: 'green', 1: 'red'}, multiple='stack')
plt.savefig(os.path.join(visuals_folder, "habitat_hist.png"))
plt.show()

This snippet of code generates a histogram to visualize the distribution of the habitat feature with respect to mushroom edibility:

###### i) Figure Size:

plt.figure(figsize=(14,2))
Sets the size of the figure to 14 inches wide by 2 inches tall. This can be adjusted depending on how you want the visualization to appear.

###### ii) Create Histogram:

sns.histplot(data=df, x='habitat', hue='poisonous', palette={0: 'green', 1: 'red'}, multiple='stack')

data=df: Specifies the DataFrame containing the data.

x='habitat': Indicates that the histogram is based on the habitat column.

hue='poisonous': Adds color to the histogram based on the poisonous column (0 for edible, 1 for poisonous).

palette={0: 'green', 1: 'red'}: Assigns green to edible mushrooms and red to poisonous ones.

multiple='stack': Stacks the histogram bars for each category (edible and poisonous) on top of each other to show their distribution within each habitat category.

###### iii) Save the Plot:

plt.savefig(os.path.join(visuals_folder, "habitat_hist.png"))
Saves the figure to the specified path in visuals_folder with the filename "habitat_hist.png".

###### iv) Display the Plot:
plt.show()
Displays the plot in the output.


This same snippet of code generates a histogram to visualize the distribution of the other 21 features with respect to mushroom edibility:

Thus we have generated 22 histograms for all the columns namely 
1) cap-shape' 2) 'cap-surface' 3) 'cap-color' 4) 'bruises' 5) 'odor'
6) 'gill-attachment' 7) 'gill-spacing' 8)'gill-size' 9) 'gill-color'
10) 'stalk-shape' 11) 'stalk-root' 12) 'stalk-surface-above-ring'
13) 'stalk-surface-below-ring' 14) 'stalk-color-above-ring'
15) 'stalk-color-below-ring' 16) 'veil-type' 17) 'veil-color' 18) 'ring-number'
19) 'ring-type' 20) 'spore-print-color' 21) 'population' 22) 'habitat'

These histograms help visualize how the distribution of these variables varies between edible and poisonous mushrooms.

""" for col in df.columns:

plt.figure(figsize=(15,2))
sns.histplot(data=df, x = col, hue='poisonous', palette={0: 'green', 1: 'red'}, multiple='stack')
plt.show()
"""

I could have used the above commented loop to produce these visualizations but it wouldn't have been possible to comment on each visual.

#### D) Data Cleaning (handling null values)

##### 1) Checking for Null Values
df.isna().sum() / len(df) * 100
This line calculates the percentage of missing (NaN) values in each column of the DataFrame df.

df.isna().sum() counts the number of missing values per column.

Dividing by len(df) (the total number of rows) and multiplying by 100 gives the percentage of missing values for each column.

##### 2) Imputation of Missing Values in stalk-root Column

I identified that 30% of the stalk-root column contains missing values and decided to impute (fill) these with the most frequent value (the mode):

df['stalk-root'] = df['stalk-root'].fillna(df['stalk-root'].mode()[0])

df['stalk-root'].fillna() replaces the NaN values in the stalk-root column.

df['stalk-root'].mode()[0] gets the mode (most frequent value) of the stalk-root column.

This effectively replaces all missing values with the most common value of that column.

##### 3) Verify that Missing Values are Handled
df.isna().sum() / len(df) * 100

This line checks again for missing values, confirming that there are no more NaN values in the DataFrame after imputation.

##### 4) Describing the Target Variable

df.describe()
This provides descriptive statistics for the DataFrame df, including count, mean, standard deviation, and percentiles. It helps in understanding the distribution of the numerical features, including the target variable poisonous.

##### 5) Creating a Copy of the DataFrame
df_copy = df.copy()

This creates a deep copy of the df DataFrame and stores it in df_copy. A deep copy means that changes made to the original df will not affect df_copy and vice versa.

You create this copy for comparison purposes later, allowing you to safely manipulate the original data without losing the ability to revert or compare.

###### 6) Verifying the Copy
df_copy.head()

Displays the first 5 rows of df_copy to verify that the copy was created correctly.

#### E) Encoding and Scaling the dataset (further Feature Engineering)¶

##### 1) Identifying Numerical and Categorical Columns

def data_type(dataset):
    numerical = []
    categorical = []
    for i in dataset.columns:
        if dataset[i].dtype == 'int64' or dataset[i].dtype == 'float64' or dataset[i].dtype == 'int8':
            numerical.append(i)
        else:
            categorical.append(i)
    return numerical, categorical

The function data_type is used to separate numerical and categorical columns from the dataset based on their data types (int64, float64, etc. for numerical).

We check the numerical and categorical lists after calling this function.

##### 2) Encoding Categorical Columns

def encoding(dataset, categorical):
    for i in categorical:
        dataset[i] = dataset[i].astype('category')
        dataset[i] = dataset[i].cat.codes
    return dataset

This function converts categorical columns to integer codes using .cat.codes, which replaces categories with corresponding numerical codes. This step is necessary for machine learning models that require numeric input.

We call the data_type function again to update the lists numerical and categorical.

##### 3) Identifying Binary Columns

def binary_columns(dataset):
    binary_cols = []
    for col in dataset.select_dtypes(include=['int', 'float']).columns:
        unique_values = dataset[col].unique()
        if np.in1d(unique_values, [0, 1]).all():
            binary_cols.append(col)
    return binary_cols

The function binary_columns identifies columns with only binary values (0 and 1). These columns are excluded from scaling since binary data should not be transformed.

##### 4) Scaling Numerical Features

def feature_scaling(dataset, numerical):
    sc_x = StandardScaler()
    dataset[numerical] = sc_x.fit_transform(dataset[numerical])
    return dataset

The function feature_scaling applies StandardScaler to the numerical columns, standardizing them by centering them around the mean and scaling to unit variance. This ensures all numerical features are on the same scale for model training.

##### 5) Checking the encoded and scaled data

dataset.tail():
This displays the last 5 rows of the dataset DataFrame.
It is typically used to inspect the data at the end of the DataFrame to ensure everything looks correct after the previous operations (e.g., encoding, scaling, etc.).

##### 6) Dropping 'veil-type' column

dataset['veil-type'].unique():

This checks for the unique values in the 'veil-type' column of the dataset.
It is used to determine whether the column contains multiple unique values or just a single unique value. 
In this case, it’s checking if 'veil-type' contains only one distinct value across all rows.

dataset = dataset.drop('veil-type', axis=1):

This removes the 'veil-type' column from the DataFrame.
The reason for dropping this column is because, based on the previous check, it likely contains only one unique value (i.e., it doesn't provide any useful information for classification). 

Columns with a single unique value do not contribute to model learning, as they don't offer variability or help distinguish between different classes.
axis=1 indicates that you are dropping a column (if it were axis=0, you'd be dropping rows).

This process helps clean the dataset by removing a redundant column, which simplifies the model and reduces unnecessary computation.

##### 7) Dropping Columns with High Correlation

cormatrix = dataset.corr()
sns.heatmap(cormatrix, annot=True)

After encoding and scaling, a heatmap of the correlation matrix is plotted to identify any features with high correlations. 

The figure (a correlation matrix heatmap) is saved as an image file and then displayed on the screen or interface. This allows you to both preserve the figure and view it interactively.

If two features are highly correlated, one can be removed to avoid redundancy. The columns veil-color and gill-attachment had high positive correlations, so veil-color was dropped.

After dropping the veil-color column, the heatmap of the correlation matrix is plotted again to verify if there are no more high corrrelations.

The figure (a correlation matrix heatmap) is again saved as an image file and then displayed on the screen or interface. This allows you to both preserve the figure and view it interactively.


##### 8) Saving Processed Data

file_path = os.path.join("C:/Users/nikde/Documents/UpGrad/mushroomclassification/data/processeddata", "processed_data.csv")
dataset.to_csv(file_path, index=False)

The cleaned and preprocessed dataset is saved as a CSV file for future use in model training.

#### F) Performing Train-Test split on the dataset

This code performs a train-test split on the dataset:

##### 1) Extracting independent variables (x):

x = dataset.iloc[:,:20].values: 

This selects the first 20 columns of the dataset (i.e., all features except the target variable) and stores them in x. The .values converts the DataFrame into a NumPy array.

##### 2) Extracting dependent variable (y):

y = dataset.iloc[:,[-1]].values: This selects the last column of the dataset (the target variable) and stores it in y as a NumPy array.

##### 3) Train-test split:

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45):
This splits the data into training and testing sets.

test_size=0.2 means 20% of the data will be used for testing, and 80% will be used for training.

random_state=45 ensures that the split is reproducible. Every time you run the code with this random state, the same split will occur.

##### 4) Checking the dimensions of the splits:

X_train.shape, y_train.shape: These commands print the dimensions of the training sets.

X_test.shape, y_test.shape: These commands print the dimensions of the testing sets.

This helps confirm that the split was performed correctly and that the shape of the train and test sets are as expected.

Finally, the X_train, X_test, y_train, and y_test variables are printed to verify the contents of each.

#### G) Model Training and Evaluation

In this code, I am training a Logistic Regression model on my dataset and evaluating its performance using various metrics

##### 1) Model Training:

logmodel_ini = LogisticRegression()
logmodel_ini.fit(X_train, y_train)

You initialize a Logistic Regression model and fit it on the training data (X_train and y_train).

##### 2) Making Predictions:

y_pred_ini = logmodel_ini.predict(X_test)

The model is used to make predictions on the test data (X_test), and the predictions are stored in y_pred_ini.

##### 3) Accuracy:

accuracy_score(y_test, y_pred_ini)

This calculates the accuracy of the model on the test set, which measures the percentage of correct predictions.

##### 4) Confusion Matrix:

cm = confusion_matrix(y_test, y_pred_ini)

The confusion matrix is created to show the number of true positives, true negatives, false positives, and false negatives.
The ConfusionMatrixDisplay is used to plot and visualize the confusion matrix.

##### 5) ROC Curve:

fpr, tpr, thresholds = roc_curve(y_test, y_pred_ini)
roc_auc = auc(fpr, tpr)

The Receiver Operating Characteristic (ROC) curve is generated to show the performance of the model at different classification thresholds. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR).

The Area Under the Curve (AUC) is calculated to summarize the overall performance of the ROC curve. A higher AUC value indicates better model performance.

##### 6) F1 Score:

print(classification_report(y_test, logmodel_ini.predict(X_test), target_names=target_names))

The F1 score, precision, and recall are calculated and displayed in a classification report. 

The F1 score is the harmonic mean of precision and recall, making it a good metric for imbalanced datasets.

##### 7) Log Loss:

log_loss(y_test, logmodel_ini.predict(X_test))

Logarithmic Loss (Log Loss) measures the performance of a classification model by penalizing false classifications. A lower value indicates a better model.

##### 8) Training and Testing Accuracy:

You check the accuracy of the model on both the training and test sets:
accuracy_score(y_train, y_pred_train)  # Training accuracy
accuracy_score(y_test, y_pred_ini)  # Testing accuracy

This gives you a comprehensive evaluation of the Logistic Regression model's performance across several metrics and plots, such as the confusion matrix, ROC curve, accuracy, F1 score, and log loss.

#### H) Applying K-Fold Cross Validation to perform Hyperparameter tuning to find the best value of C (C = 1/Lambda)

In this code, I'm using K-Fold Cross Validation to perform hyperparameter tuning for the regularization parameter 𝐶
in a Logistic Regression model. Here's a breakdown of the key steps:

##### 1) Define C values for tuning:

C = [0.01, 0.1, 1, 10, 100, 1000]

These are the different values of 
𝐶(regularization strength) you want to try. 

##### 2) Initialize a loop to perform cross-validation:

for c in C:
    logmodel = LogisticRegression(C=c)
    scores = cross_val_score(logmodel, X_train, y_train, cv=3, scoring='accuracy')
    cv_score.append(scores.mean())
For each value of 𝐶, 
you initialize a new LogisticRegression model and apply 3-fold cross-validation using the cross_val_score method.

The cross-validation accuracy scores for each fold are averaged and stored in cv_score.

##### 3) Finding the best value of C:

best_index = np.argmax(cv_score)
best_C = C[best_index]
print(f"The best value of C is: {best_C}")

The index of the best 𝐶 value is found using np.argmax(), which returns the index of the maximum value in cv_score.
The best 𝐶 is printed based on this index.

##### 4) Plotting Cross-Validation Accuracy:

plt.plot(C, cv_score)
plt.xscale('log')  # Use log scale for better readability
plt.xlabel('C (Regularization Strength)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Cross-Validation Accuracy for Different C Values')

You plot the cross-validation accuracy scores against the corresponding 
𝐶 values. 

The x-axis is in logarithmic scale (plt.xscale('log')) for better visualization since 
𝐶 values span several orders of magnitude.

##### 5) Save the plot:

plt.savefig(os.path.join(visuals_folder, "Cross-Validation Accuracy for Different C Values.png"))
plt.show()

The plot is saved as an image file in the specified visuals_folder.

##### 6) Output:
The best 𝐶 value for your model, based on the highest cross-validation accuracy, is printed and visualized through the plot. This helps you choose the optimal regularization strength for your Logistic Regression model.

#### I) Using different regularization models to find the best performing model

In this part of MY workflow, I am applying three different types of regularization models for Logistic Regression: 
L2 (Ridge), L1 (Lasso), and Elastic Net (L1 + L2), to compare their performances on both the training and test datasets. 

Here's a breakdown of each model and the steps:

##### 1) L2 Regularization (Ridge) Model

Logistic Regression with L2 Regularization applies penalty on the square of the magnitude of coefficients 
to prevent overfitting and manage multicollinearity.

LogisticRegression(C=10, penalty='l2', solver='liblinear'): Initializes a logistic regression model with L2 regularization (ridge). 

C is the inverse of the regularization strength; smaller values mean stronger regularization. 

penalty='l2' specifies L2 regularization, and solver='liblinear' is used for small datasets.

fit(X_train, y_train): Trains the model using the training data.

coef_: Gets the coefficients of the features from the trained model.

predict(X_test): Makes predictions on the test set.

Accuracy Calculation :

accuracy_score(y_test, y_pred_ridge): Computes the accuracy of the model on the test dataset.

predict(X_train): Predicts on the training dataset to check if the model overfits.

accuracy_score(y_train, y_pred_train_ridge): Computes the accuracy on the training dataset.

##### 2) L1 Regularization (Lasso) Model

penalty='l1': Specifies L1 regularization (lasso). This can lead to sparse solutions where some feature coefficients are zero.

The rest of the code is similar to the L2 model for training and predicting.

Accuracy Calculation:

accuracy_score(y_test, y_pred_lasso): Computes the Accuracy on the test set.

accuracy_score(y_train, y_pred_train_lasso): Computes the Accuracy on the training set.

##### 3) Elastic Net (L1 + L2) Regularization Model

penalty='elasticnet': Uses a combination of L1 and L2 regularization. The l1_ratio parameter controls the mix between L1 and L2 regularization.

l1_ratio=0.5: Indicates that the penalty is equally divided between L1 and L2 regularization.

solver='saga': Required for elastic net regularization and works well with large datasets.

Accuracy Calculation:

accuracy_score(y_test, y_pred_elasticnet): Computes the Accuracy on the test set.

accuracy_score(y_train, y_pred_train_elasticnet): Computes the Accuracy on the training set.

##### 4) Summary
L2 (Ridge) Regularization: Useful for reducing model complexity and preventing overfitting by shrinking the coefficients.

L1 (Lasso) Regularization: Useful for feature selection as it can shrink some coefficients to zero, effectively removing them from the model.

Elastic Net Regularization: Combines both L1 and L2 regularization to balance feature selection and coefficient shrinking.

By comparing the accuracy scores from these different models, you can choose the regularization technique that best fits your data and model performance requirements.

##### 5) Define the Folder

model_folder = "C:/Users/nikde/Documents/UpGrad/mushroomclassification/models"

model_folder: Specifies the path where the models will be saved. Ensure this path exists or create it before saving models.

##### 6) Save Each Model
joblib.dump(logmodel_ini, os.path.join(model_folder, "Initial_model.pkl"))
joblib.dump(logmodel_ridge, os.path.join(model_folder, "L2_Ridge_Regression_model.pkl"))
joblib.dump(logmodel_lasso, os.path.join(model_folder, "L1_Lasso_Regression_model.pkl"))
joblib.dump(logmodel_elasticnet, os.path.join(model_folder, "L1_L2_ElasticNet_model.pkl"))

joblib.dump: This function serializes and saves the Python object (in this case, a trained model) to a file.

os.path.join(model_folder, "filename.pkl"): Combines the folder path with the file name to create the full path where each model will be saved.

"Initial_model.pkl": This would be the file name for the initial model.
"L2_Ridge_Regression_model.pkl": File name for the L2 regularization model.
"L1_Lasso_Regression_model.pkl": File name for the L1 regularization model.
"L1_L2_ElasticNet_model.pkl": File name for the Elastic Net regularization model.

##### 7) Confirmation Message

print("All models saved successfully!"): Prints a message to confirm that all models have been saved.

#### J) Creating the score-card and defining a function to update it with important perfromance metrics

This code snippet helps in comparing different machine learning models by evaluating their performance and storing the results in a DataFrame. Here’s a detailed explanation:

##### 1) Create an Empty DataFrame

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd

score_card = pd.DataFrame(columns=['model_name','Accuracy Score','Precision Score','Recall Score','AUC Score','f1 Score'])

pd.DataFrame(columns=[...]): Initializes an empty DataFrame with the specified column names to store the performance metrics of various models.

##### 2) Define the Function to Update the Scorecard

def update_score_card(y_test, y_pred, model_name):
    # Assign 'score_card' as global variable
    global score_card

    # Append the results to the DataFrame 'score_card'
    score_card = pd.concat([score_card, pd.DataFrame([{'model_name': model_name,
                                        'Accuracy Score': accuracy_score(y_test, y_pred),
                                        'Precision Score': precision_score(y_test, y_pred),
                                        'Recall Score': recall_score(y_test, y_pred),
                                        'AUC Score': roc_auc_score(y_test, y_pred),
                                        'f1 Score': f1_score(y_test, y_pred)}])],
                                        ignore_index=True)

global score_card: Declares score_card as a global variable so that the function can modify it.

pd.concat([...], ignore_index=True): Concatenates the new metrics into the existing score_card DataFrame without considering the old index. The new index is reset.

##### 3) Call the Function for Each Model

update_score_card(y_test, y_pred_ini, 'initial_model'): Updates the scorecard with metrics for the initial model.

update_score_card(y_test, y_pred_ridge, 'Ridge Regression - L2 Reg'): Updates the scorecard with metrics for the Ridge Regression model.

update_score_card(y_test, y_pred_lasso, 'Lasso Regression - L1 Reg'): Updates the scorecard with metrics for the Lasso Regression model.

update_score_card(y_test, y_pred_elasticnet, 'Elastic Net - L1 and L2'): Updates the scorecard with metrics for the Elastic Net model.

score_card: Displays the final scorecard with all the metrics for comparison.

##### 4) Summary
Metrics Computed: Accuracy, Precision, Recall, AUC (Area Under the Curve), and F1 Score.

update_score_card Function: Adds performance metrics of a model to the scorecard.

Comparing Models: By calling this function for different models, you can easily compare their performance.

This approach provides a clear, tabulated view of how different models perform on the same dataset, which is useful for model evaluation and selection.
