

wrkdata = fread('Churn_Modelling.csv', stringsAsFactors = F)

# Convert Factors
wrkdata$Geography = as.numeric(as.factor(wrkdata$Geography))
wrkdata$Gender = as.numeric(as.factor(wrkdata$Gender))

# Set up training/test sets
x = wrkdata[, 4:13]
y = wrkdata[, 14]

library(caTools)
split = sample.split(1:nrow(y), SplitRatio = 0.8)
x.train = scale(x[split])
y.train = as.matrix(y[split])

x.test = scale(x[!split])
y.test = as.matrix(y[!split])

# Build ANN
model = keras_model_sequential() %>%
        layer_dense(units = 6, activation = 'relu', input_shape = 10) %>%
        layer_dense(units = 6, activation = 'relu') %>%
        layer_dense(units = 1, activation = 'sigmoid')

model %>% compile(loss = "mse", optimizer = optimizer_rmsprop(),
                  metrics = list("accuracy"))

history = model %>% fit(x.train, y.train, epochs = 25)
plot(history,Metrics='accuracy')

z = model %>% predict(x.test)

# ==========

boston_housing <- dataset_boston_housing()

c(train_data, train_labels) %<-% boston_housing$train
c(test_data, test_labels) %<-% boston_housing$test
paste0("Training entries: ", length(train_data), ", labels: ", length(train_labels))

column_names <- c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                  'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT')
train_df <- as_tibble(train_data)
colnames(train_df) <- column_names

train_data <- scale(train_data)

# Use means and standard deviations from training set to normalize test set
col_means_train <- attr(train_data, "scaled:center")
col_stddevs_train <- attr(train_data, "scaled:scale")
test_data <- scale(test_data, center = col_means_train, scale = col_stddevs_train)

build_model <- function() {

    model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[2]) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dense(units = 1)

    model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(),
    metrics = list("accuracy")
  )

    model
}

model <- build_model()
model %>% summary()

epochs <- 100

# Fit the model and store training stats
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  verbose = 1,
  #callbacks = list(print_dot_callback)
)

library(ggplot2)

plot(history, metrics = "accuracy", smooth = FALSE) +
    coord_cartesian(ylim = c(0, 5))