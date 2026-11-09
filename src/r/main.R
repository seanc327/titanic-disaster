library(readr)

print("Loading training data:")

train <- read_csv("../data/train.csv", show_col_types = FALSE)
# for import errors since i kept getting them
if (nrow(train) == 0 || ncol(train) == 0) {
  stop("Training data empty.")
}

print(paste("Number of rows:", nrow(train)))
print(paste("Number of columns:", ncol(train)))

print("First five rows:")
print(head(train, 5), width = Inf)

# repeat 14-21
# 14
# data exploration
print("Data Exploration:")

print("Missing values before preparation:")
print(colSums(is.na(train)))

print("Survival counts (0 = died, 1 = survived):")
print(table(train$Survived))

# features
print("Add features:")
train$FamilySize <- train$SibSp + train$Parch + 1
print("Added FamilySize = SibSp + Parch + 1 (includes the passenger).")

# data adjustment
print("Data adjustment:")

features <- c("Pclass", "Sex", "Age", "FamilySize", "Fare")
train_model <- as.data.frame(train[, c("Survived", features)])

print("Selected features:")
print(features)
print("Target: Survived")
print("Excluded PassengerId, Name, Ticket, Cabin, and Embarked.")

train_model$Sex <- ifelse(train_model$Sex == "male", 0, 1)
print("Converted Sex to binary numbers: male = 0, female = 1.")

age_median <- median(train_model$Age, na.rm = TRUE)
missing_ages <- sum(is.na(train_model$Age))
train_model$Age[is.na(train_model$Age)] <- age_median

print(paste("Filled", missing_ages, "missing Age values with", age_median))

print("Missing values after preparation:")
print(colSums(is.na(train_model)))

# 15 log regression model to predict survivability on training set
print("Building logistic regression model to predict survivability on training set:")

model <- glm(
  Survived ~ Pclass + Sex + Age + FamilySize + Fare,
  data = train_model,
  family = binomial(),
  control = glm.control(maxit = 1000)
)

print(paste(
  "Trained model on", nrow(train_model),
  "passengers, using logistic regression with maxit = 1000."
))
print(paste("Model converged:", model$converged))

# 16 training accuracy
print("Training accuracy:")

train_probabilities <- predict(model, type = "response")
train_predictions <- ifelse(train_probabilities >= 0.5, 1, 0)
print("Converted survival probabilities to predictions using a 0.5 threshold.")

correct <- sum(train_predictions == train_model$Survived)
train_accuracy <- mean(train_predictions == train_model$Survived)

print(paste("Correct predictions:", correct, "out of", nrow(train_model)))
print(sprintf("Training accuracy: %.2f%%", train_accuracy * 100))

# 17
print("Test data predictions:")

test <- read_csv("../data/test.csv", show_col_types = FALSE)
# test to check emptiness
if (nrow(test) == 0) {stop("Test data empty.")}

print(paste("Number of rows:", nrow(test)))
print(paste("Number of columns:", ncol(test)))

# prepare test data like how we did train
print("Preparing test data:")

test$FamilySize <- test$SibSp + test$Parch + 1
print("Added FamilySize = SibSp + Parch + 1.")

test_model <- as.data.frame(test[, features])
print("Using same features as training:")
print(features)

test_model$Sex <- ifelse(test_model$Sex == "male", 0, 1)
print("Converted Sex to binary: male = 0, female = 1.")

# missing data
missing_test_ages <- sum(is.na(test_model$Age))
test_model$Age[is.na(test_model$Age)] <- age_median
print(paste("Filled", missing_test_ages, "missing ages with training median:", age_median))

fare_median <- median(train_model$Fare, na.rm = TRUE)
missing_test_fares <- sum(is.na(test_model$Fare))
test_model$Fare[is.na(test_model$Fare)] <- fare_median
print(paste("Filled", missing_test_fares, "missing fares with training median:", fare_median))

print("Missing values after test preparation:")
print(colSums(is.na(test_model)))

# predictions
test_probabilities <- predict(model, newdata = test_model, type = "response")
test_predictions <- ifelse(test_probabilities >= 0.5, 1, 0)
print("Applied the trained model with a 0.5 prediction threshold.")

predictions <- data.frame(PassengerId = test$PassengerId, Survived = test_predictions)

print("First ten predictions (0 = dead, 1 = survived):")
print(head(predictions, 10), row.names = FALSE)

print(paste("Generated predictions for", nrow(predictions), "passengers."))
print("Prediction counts:")
print(table(predictions$Survived))

# 18
print("Test accuracy:")
print("Since there are no survival labels, we cannot calculate accuracy for test.csv.")
print("Instead, evaluating on a held-out validation set from train.csv.")

# start with features before missing ages were filled
print("Validation preparation:")

validation_source <- as.data.frame(train[, c("Survived", features)])
print("Used the same features, including FamilySize.")

validation_source$Sex <- ifelse(validation_source$Sex == "male", 0, 1)
print("Encoded Sex to binary: male = 0, female = 1.")

# stratified split
set.seed(42)

fit_indices <- unlist(lapply(
  split(seq_len(nrow(validation_source)), validation_source$Survived),
  function(indices) sample(indices, floor(0.8 * length(indices)))
))

fit_data <- validation_source[fit_indices, ]
valid_data <- validation_source[-fit_indices, ]

print(paste("Split labeled data:", nrow(fit_data), "training,", nrow(valid_data), "validation."))
print("Used a stratified 80/20 split with seed = 42.")

# missing data
validation_age_median <- median(fit_data$Age, na.rm = TRUE)
fit_missing <- sum(is.na(fit_data$Age))
valid_missing <- sum(is.na(valid_data$Age))

fit_data$Age[is.na(fit_data$Age)] <- validation_age_median
valid_data$Age[is.na(valid_data$Age)] <- validation_age_median

print(paste(
  "Filled", fit_missing, "training ages and", valid_missing,
  "validation ages with training-split median:", validation_age_median
))

# model
validation_model <- glm(
  Survived ~ Pclass + Sex + Age + FamilySize + Fare,
  data = fit_data,
  family = binomial(),
  control = glm.control(maxit = 1000)
)

print("Trained a separate logistic regression model on the training split only.")
print(paste("Model converged:", validation_model$converged))

valid_probabilities <- predict(validation_model, newdata = valid_data, type = "response")
valid_predictions <- ifelse(valid_probabilities >= 0.5, 1, 0)
print("Converted survival probabilities to predictions using a 0.5 threshold.")

correct_valid <- sum(valid_predictions == valid_data$Survived)
valid_accuracy <- mean(valid_predictions == valid_data$Survived)

print(paste("Correct validation predictions:", correct_valid, "out of", nrow(valid_data)))
print(sprintf("Validation accuracy: %.2f%%", valid_accuracy * 100))