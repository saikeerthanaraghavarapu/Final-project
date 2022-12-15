# install.packages("lubridate")
# install.packages("ggcorrplot")
# install.packages("glmnet")
library(plyr)
library(dplyr)
library(lubridate)
library(ggcorrplot)
library(stringr)
library(glmnet)
library(caret)

# loading dataset
print("Loading Datasets")
trainingDataset <- read.csv("./train.csv", na.strings=c("", "NA"), header=TRUE) 
testingDataset <- read.csv("./test.csv", na.strings=c("", "NA"), header=TRUE) 


# displaying top-5 rows and properties of the dataset
print("Top-5 rows of the dataset")
print(head(trainingDataset, 5))

cat("\nThe number of rows and columns in the dataset:",nrow(trainingDataset), "and", ncol(trainingDataset))

cat('\nDescriptive analysis of the dataset')
print(summary(trainingDataset))

# preprocessing datasets
# checking for the missing values
preprocessDataframe <-sapply(trainingDataset, function(y) sum(length(which(is.na(y)))))
cat("checking for the missing values")
print(preprocessDataframe)

trainingDataset$Dependents <- revalue(trainingDataset$Dependents, c("3+"="3")) 

#correlation of the datasets
corr <- round(cor(trainingDataset[sapply(trainingDataset, function(x) is.numeric(x))], use ='pairwise.complete.obs'), 1)
p <- ggcorrplot(corr, method = "circle")
print(p)

cat("Fixing null values for both the datasets")
trainingDataset['Gender'][is.na(trainingDataset['Gender'])] <- "Male"
trainingDataset['Married'][is.na(trainingDataset['Married'])] <- "yes"
trainingDataset['Dependents'][is.na(trainingDataset['Dependents'])] <- "0"
trainingDataset['Self_Employed'][is.na(trainingDataset['Self_Employed'])] <- "No"
trainingDataset['LoanAmount'][is.na(trainingDataset['LoanAmount'])] <- "0"
trainingDataset['Loan_Amount_Term'][is.na(trainingDataset['Loan_Amount_Term'])] <- "0"
trainingDataset['Credit_History'][is.na(trainingDataset['Credit_History'])] <- "0"

# checking for the missing values
preprocessDataframe <-sapply(trainingDataset, function(y) sum(length(which(is.na(y)))))
cat("checking for the missing values post processing")
print(preprocessDataframe)

# plotting histogram of the house
par(mfrow = c(2, 3))
for(i in list('ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'))
  hist(as.numeric(trainingDataset[[i]]), main = i, xlab = "Value")

cat("performing chi-square test between loan status and other attributes")
nums <- unlist(lapply(trainingDataset, is.numeric), use.names = FALSE)
df <- trainingDataset[ , nums]
print(chisq.test(df))

# dropping unwanted columns
col <- list('Loan_Amount_Term', 'Loan_ID')

dataset <- trainingDataset[, -which(names(trainingDataset) %in% col)]
cat("Top-5 rows of the dataset")
print(head(dataset, 5))


# label encoding the dataset
cols = list('Gender','Married','Education','Self_Employed','Property_Area','Loan_Status','Dependents')
for (col in cols)
  dataset[col] <- as.numeric(factor(dataset[,col]))


print("Segregating inputs and outputs")

Y = dataset[c('LoanAmount')]
X = dataset[,-c(8)]

print("Sample from input and output")
print(head(X, 5))
print(head(Y, 5))

# Splitting dataset into train and test sets
set.seed(5)
# 70% of the sample size
train_size <- floor(0.75 * nrow(dataset))

in_rows <- sample(c(1:nrow(dataset)), size = train_size, replace = FALSE)

X_train <- X[in_rows, ]
X_test <- X[-in_rows, ]
y_train <- Y[in_rows, ]
y_test <- Y[-in_rows, ]

cat("\nThe number of rows and columns in the train dataset:",nrow(X_train), "and", ncol(X_train))
cat("\nThe number of rows and columns in the test dataset:",nrow(X_test), "and", ncol(X_test))

knnmodel <- knnreg(X_train, as.numeric(y_train))

cat("creating KNN model and training the model with the training dataset\n")

cat("Summary of KNN Model\n")
print(knnmodel)
print(str(knnmodel))

cat("Predicting loan amount for the test dataset\n")
pred_y = predict(knnmodel, data.frame(X_test))

cat("Accuracy checking\n")
results <- data.frame(as.numeric(y_test), pred_y)
print(head(results, 5))


mse = mean((as.numeric(y_test) - pred_y)^2)
mae = MAE(as.numeric(y_test), pred_y)
rmse = RMSE(as.numeric(y_test), pred_y)

cat("MSE: ", mse, "MAE: ", mae, " RMSE: ", rmse)

plot.new()
dev.new(width = 15,  
        height = 3)
x = 1:length(as.numeric(y_test))

plot(x, as.numeric(y_test), col = "red", type = "l", lwd=2,
     main = "Loan Prediction test data prediction")
lines(x, pred_y, col = "blue", lwd=2)
legend("topright",  legend = c("original", "predicted"), 
       fill = c("red", "blue"), col = 2:3,  adj = c(0, 0.6))
grid()

