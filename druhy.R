library(readr)
library(keras)
library(dplyr)
library(tfhub)

#### READING INPUTS ####

raw.train <- read_csv("oie-train.csv")
raw.dev <- read_csv("oie-devin.csv")
raw.test <- read_csv("oie-testin.csv")

#### PREPARING EMBEDDING LAYER ####
lembd <- layer_hub(handle = "https://tfhub.dev/google/universal-sentence-encoder/4")

emb.train.prem <- lembd(raw.train$Premise)
emb.train.hyp <- lembd(raw.train$Hypothesis)
emb.dev.prem <- lembd(raw.dev$Premise)
emb.dev.hyp <- lembd(raw.dev$Hypothesis)
emb.test.prem <- lembd(raw.test$Premise)
emb.test.hyp <- lembd(raw.test$Hypothesis)

matrix.train.prem <- as.matrix(emb.train.prem)
matrix.train.hypo <- as.matrix(emb.train.hyp)
matrix.dev.prem <- as.matrix(emb.dev.prem)
matrix.dev.hypo <- as.matrix(emb.dev.hyp)
matrix.test.prem <- as.matrix(emb.test.prem)
matrix.test.hypo <- as.matrix(emb.test.hyp)

#### LABELS PREPARATION ####

make.labels <- function(clss) {
  len <- length(clss)
  res <- matrix(0, nrow = len, ncol = 3)
  res[clss=="entailment", 1] <- 1
  res[clss=="neutral", 2] <- 1
  res[clss=="contradiction", 3] <- 1
  return(res)
}

train.labels <- make.labels(raw.train$GoldLabel)
dev.labels <- make.labels(raw.dev$GoldLabel)
test.labels <- make.labels(raw.test$GoldLabel)

#### MODEL ####
input1 <- layer_input(shape = 512)
input2 <- layer_input(shape = 512) 

tuning <- layer_dense(units = 92)

ti1 <- input1 %>% tuning
ti2 <- input2 %>% tuning

lb <- layer_concatenate(list(ti1, ti2)) %>% layer_dense(48, activation = "relu") %>% layer_dropout(0.3) %>% layer_dense(16, activation = 'relu') %>% layer_dense(3, activation = "softmax")

model <- keras_model(inputs = list(input1, input2), outputs = lb)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)


# training
model %>%
  fit(
    list(matrix.train.prem, matrix.train.hypo),
    train.labels,
    batch_size = 64L,
    epochs = 32,
    validation_data = list(
      list(
        matrix.dev.prem,
        matrix.dev.hypo
      ),
      dev.labels)
  )

#### testing

evalres <- evaluate(model, x = list(matrix.test.prem, matrix.test.hypo), y=test.labels)
evalres
