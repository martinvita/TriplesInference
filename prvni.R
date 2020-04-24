library(readr)
library(keras)
library(dplyr)
raw.train <- read_csv("oie-train.csv")
raw.dev <- read_csv("oie-devin.csv")
raw.test <- read_csv("oie-testin.csv")

FLAGS <- flags(
  flag_integer("vocab_size", 50000),
  flag_integer("max_len_padding", 3),
  flag_integer("embedding_size", 300)
)

# tokenizer from Keras
tokenizer <- text_tokenizer(num_words = FLAGS$vocab_size)

train.prem.subj <- raw.train$PremiseSubj
train.prem.rel <- raw.train$PremiseRel
train.prem.obj <- raw.train$PremiseObj

train.hyp.subj <- raw.train$HypothesisSubj
train.hyp.rel <- raw.train$HypothesisRel
train.hyp.obj <- raw.train$HypothesisObj

dev.prem.subj <- raw.dev$PremiseSubj
dev.prem.rel <- raw.dev$PremiseRel
dev.prem.obj <- raw.dev$PremiseObj

dev.hyp.subj <- raw.dev$HypothesisSubj
dev.hyp.rel <- raw.dev$HypothesisRel
dev.hyp.obj <- raw.dev$HypothesisObj

test.prem.subj <- raw.test$PremiseSubj
test.prem.rel <- raw.test$PremiseRel
test.prem.obj <- raw.test$PremiseObj

test.hyp.subj <- raw.test$HypothesisSubj
test.hyp.rel <- raw.test$HypothesisRel
test.hyp.obj <- raw.test$HypothesisObj

txt.compl <- c(train.prem.subj, train.prem.rel, train.prem.obj,
               train.hyp.subj, train.hyp.rel, train.hyp.obj,
               
               dev.prem.subj, dev.prem.rel, dev.prem.obj,
               dev.hyp.subj, dev.hyp.rel, dev.hyp.obj,
               
               test.prem.subj, test.prem.rel, test.prem.obj,
               test.hyp.subj, test.hyp.rel, test.hyp.obj)


fit_text_tokenizer(tokenizer, txt.compl)

# in different versions of Keras take care about word_index vs. index_word
word.index <- tokenizer$word_index
voc.size <- min(length(word.index), FLAGS$vocab_size)
words <- names(word.index)[1:voc.size]

# first steps in constructing lookup table: index;word
aux.index.table <- data.frame(Indx=1:voc.size, Wrd=words, stringsAsFactors = F)

# we will use Glove embeddings available within http://nlp.stanford.edu/data/glove.6B.zip archive
glove <- read_table2("glove.6B.300d.txt", col_names = F, skip = 0, progress = T)

# first column in glove dataframe will be called "Wrd"
colnames(glove) <- c("Wrd", paste0("S", 1:300))

# attaching glove embeddings to lookup table 
aux.join <- left_join(aux.index.table, glove, by = "Wrd")

# construction of embedding matrix - in the i-th row, there is a vector of i-th word (w.r.t. tokenization)
emb.mat <- as.matrix(aux.join[,3:ncol(aux.join)])
emb.mat[is.na(emb.mat)] <- 0
# dirty trick - problem of indexing in R vs. Python (first index in R is 1, in Python 0)
emb.mat2 <- rbind(rep(0, times = 300), emb.mat)

# text are transformed in the sequences of integers w.r.t. tokenization
s.train.prem.subj <- texts_to_sequences(tokenizer = tokenizer, train.prem.subj)
s.train.prem.rel <- texts_to_sequences(tokenizer = tokenizer, train.prem.rel)
s.train.prem.obj <- texts_to_sequences(tokenizer = tokenizer, train.prem.obj)
s.train.hyp.subj <- texts_to_sequences(tokenizer = tokenizer, train.hyp.subj)
s.train.hyp.rel <- texts_to_sequences(tokenizer = tokenizer, train.hyp.rel)
s.train.hyp.obj <- texts_to_sequences(tokenizer = tokenizer, train.hyp.obj)

s.dev.prem.subj <- texts_to_sequences(tokenizer = tokenizer, dev.prem.subj)
s.dev.prem.rel <- texts_to_sequences(tokenizer = tokenizer, dev.prem.rel)
s.dev.prem.obj <- texts_to_sequences(tokenizer = tokenizer, dev.prem.obj)
s.dev.hyp.subj <- texts_to_sequences(tokenizer = tokenizer, dev.hyp.subj)
s.dev.hyp.rel <- texts_to_sequences(tokenizer = tokenizer, dev.hyp.rel)
s.dev.hyp.obj <- texts_to_sequences(tokenizer = tokenizer, dev.hyp.obj)

s.test.prem.subj <- texts_to_sequences(tokenizer = tokenizer, test.prem.subj)
s.test.prem.rel <- texts_to_sequences(tokenizer = tokenizer, test.prem.rel)
s.test.prem.obj <- texts_to_sequences(tokenizer = tokenizer, test.prem.obj)
s.test.hyp.subj <- texts_to_sequences(tokenizer = tokenizer, test.hyp.subj)
s.test.hyp.rel <- texts_to_sequences(tokenizer = tokenizer, test.hyp.rel)
s.test.hyp.obj <- texts_to_sequences(tokenizer = tokenizer, test.hyp.obj)


# padding sequences with 0 to the same length
t.train.prem.subj <- pad_sequences(s.train.prem.subj, maxlen = FLAGS$max_len_padding, value = 0)
t.train.prem.rel <- pad_sequences(s.train.prem.rel, maxlen = FLAGS$max_len_padding, value = 0)
t.train.prem.obj <- pad_sequences(s.train.prem.obj, maxlen = FLAGS$max_len_padding, value = 0)
t.train.hyp.subj <- pad_sequences(s.train.hyp.subj, maxlen = FLAGS$max_len_padding, value = 0)
t.train.hyp.rel <- pad_sequences(s.train.hyp.rel, maxlen = FLAGS$max_len_padding, value = 0)
t.train.hyp.obj <- pad_sequences(s.train.hyp.obj, maxlen = FLAGS$max_len_padding, value = 0)

t.dev.prem.subj <- pad_sequences(s.dev.prem.subj, maxlen = FLAGS$max_len_padding, value = 0)
t.dev.prem.rel <- pad_sequences(s.dev.prem.rel, maxlen = FLAGS$max_len_padding, value = 0)
t.dev.prem.obj <- pad_sequences(s.dev.prem.obj, maxlen = FLAGS$max_len_padding, value = 0)
t.dev.hyp.subj <- pad_sequences(s.dev.hyp.subj, maxlen = FLAGS$max_len_padding, value = 0)
t.dev.hyp.rel <- pad_sequences(s.dev.hyp.rel, maxlen = FLAGS$max_len_padding, value = 0)
t.dev.hyp.obj <- pad_sequences(s.dev.hyp.obj, maxlen = FLAGS$max_len_padding, value = 0)

t.test.prem.subj <- pad_sequences(s.test.prem.subj, maxlen = FLAGS$max_len_padding, value = 0)
t.test.prem.rel <- pad_sequences(s.test.prem.rel, maxlen = FLAGS$max_len_padding, value = 0)
t.test.prem.obj <- pad_sequences(s.test.prem.obj, maxlen = FLAGS$max_len_padding, value = 0)
t.test.hyp.subj <- pad_sequences(s.test.hyp.subj, maxlen = FLAGS$max_len_padding, value = 0)
t.test.hyp.rel <- pad_sequences(s.test.hyp.rel, maxlen = FLAGS$max_len_padding, value = 0)
t.test.hyp.obj <- pad_sequences(s.test.hyp.obj, maxlen = FLAGS$max_len_padding, value = 0)

# one-hot encoding of categorical variables (we do not use to_categorical function in Keras) 
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

input.prem.subj <- layer_input(shape = c(FLAGS$max_len_padding))
input.prem.rel <- layer_input(shape = c(FLAGS$max_len_padding))
input.prem.obj <- layer_input(shape = c(FLAGS$max_len_padding))

input.hyp.subj <- layer_input(shape = c(FLAGS$max_len_padding))
input.hyp.rel <- layer_input(shape = c(FLAGS$max_len_padding))
input.hyp.obj <- layer_input(shape = c(FLAGS$max_len_padding))



embeddovaci <- layer_embedding(
  input_dim = voc.size + 1,
  output_dim = FLAGS$embedding_size,
  input_length = FLAGS$max_len_padding,
  
  # weights are stored in the embedding matrix, first row contains null vector
  weights = list(emb.mat2),
  
  # word embeddings are not trained
  trainable = F
)

subob <- layer_dense(units = 64, activation = 'relu')
relational <- layer_dense(units = 20, activation = 'relu')

# encoding of a subject of the premise is obtained by summing word embeddings, the result is fed to a dense layer (64 units)
enc.prem.subj <- input.prem.subj %>% embeddovaci %>% k_sum(axis = 2) %>% subob
enc.prem.rel <- input.prem.rel %>% embeddovaci %>% k_sum(axis = 2) %>% relational
enc.prem.obj <- input.prem.obj %>% embeddovaci %>% k_sum(axis = 2) %>% subob

enc.hyp.subj <- input.hyp.subj %>% embeddovaci %>% k_sum(axis = 2) %>% subob
enc.hyp.rel <- input.hyp.rel %>% embeddovaci %>% k_sum(axis = 2) %>% relational
enc.hyp.obj <- input.hyp.obj %>% embeddovaci %>% k_sum(axis = 2) %>% subob

# representations of all parts of both triples are concatenated  
finrep <- layer_concatenate(list(enc.prem.subj, enc.prem.rel, enc.prem.obj, enc.hyp.subj, enc.hyp.rel, enc.hyp.obj))

# concatenated representations are fed to top dense layers
predictions <- finrep  %>% layer_dense(16, activation = "relu") %>% layer_dense(3, activation = "softmax")

model <- keras_model(inputs = list(input.prem.subj, input.prem.rel, input.prem.obj, input.hyp.subj, input.hyp.rel, input.hyp.obj), outputs = predictions)

model %>% compile(
  optimizer = "rmsprop",
  loss = 'categorical_crossentropy',
  metrics = c("accuracy")
)


model %>% fit(
  list(t.train.prem.subj,
       t.train.prem.rel,
       t.train.prem.obj,
       t.train.hyp.subj,
       t.train.hyp.rel,
       t.train.hyp.obj), train.labels,
  
   epochs = 64,
   batch_size = 64,
   validation_data = list(
   list(
      t.dev.prem.subj,
      t.dev.prem.rel,
      t.dev.prem.obj,
      t.dev.hyp.subj,
      t.dev.hyp.rel,
      t.dev.hyp.obj
    ), dev.labels)
)
