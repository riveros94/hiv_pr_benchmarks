library(keras)
library(caret)
set.seed(42)
# Function to load and process sequence data
process_fasta <- function(filepath, bp) {
  fasta <- readLines(filepath)
  data <- matrix(nrow = length(fasta)/2, ncol = 2)
  for (i in seq(1, length(fasta), 2)) {
    row_idx <- (i + 1) / 2
    data[row_idx, 1] <- substr(fasta[i], nchar(fasta[i]), nchar(fasta[i]))
    data[row_idx, 2] <- fasta[i + 1]
  }
  data_labels <- as.numeric(data[, 1])
  data_seqs <- data[, 2]
  seqs_num <- array(dim = c(length(data_seqs), bp))
  for (i in 1:length(data_seqs)) {
    z <- data_seqs[i]
    seq <- unlist(strsplit(z, ""))
    for (k in 1:length(seq)) {
      seq[k] <- switch(seq[k],
                      "A" = 1, "a" = 1, "B" = 2, "b" = 2, "C" = 3, "c" = 3,
                      "D" = 4, "d" = 4, "E" = 5, "e" = 5, "F" = 6, "f" = 6,
                      "G" = 7, "g" = 7, "H" = 8, "h" = 8, "I" = 9, "i" = 9,
                      "J" = 10, "j" = 10, "K" = 11, "k" = 11, "L" = 12, "l" = 12,
                      "M" = 13, "m" = 13, "N" = 14, "n" = 14, "O" = 15, "o" = 15,
                      "P" = 16, "p" = 16, "Q" = 17, "q" = 17, "R" = 18, "r" = 18,
                      "S" = 19, "s" = 19, "T" = 20, "t" = 20, "U" = 21, "u" = 21,
                      "V" = 22, "v" = 22, "W" = 23, "w" = 23, "X" = 24, "x" = 24,
                      "Y" = 25, "y" = 25, "Z" = 26, "z" = 26, "." = 27, "#" = 28,
                      "~" = 29, "*" = 30, 0)
    }
    seqs_num[i, ] <- as.integer(seq)
  }
  data_list <- list()
  for (i in 1:nrow(seqs_num)) {
    seqi <- seqs_num[i, ]
    data_list[[i]] <- seqi
  }
  data_f <- pad_sequences(data_list, padding = "post", maxlen = bp)
  return(list(data_f = data_f, data_labels = data_labels))
}

load_and_predict <- function(model_path, fasta_path, output_prefix, n_runs = 100) {
  # Load model once outside the loop
  model <- load_model_tf(model_path)
  
  # Initialize vector to store execution times
  times <- numeric(n_runs)
  
  # Run predictions n_runs times
  for(i in 1:n_runs) {
    start_time <- Sys.time()
    
    # Process external data
    data <- process_fasta(fasta_path, bp=99)
    data_f <- data$data_f
    
    # Make predictions
    predictions <- model %>% predict(data_f)
    predicted_classes <- predictions %>% `>`(0.5) %>% k_cast("int32")
    predicted_classes <- as.numeric(predicted_classes)
    
    end_time <- Sys.time()
    times[i] <- as.numeric(difftime(end_time, start_time, units="secs"))
    
    # Store prediction from first run only
    if(i == 1) {
      first_prediction <- predicted_classes
    }
  }
  
  # Calculate statistics
  mean_time <- mean(times)
  sd_time <- sd(times)
  
  # Format results
  result <- sprintf(
    "Output file: %s\nPrediction: %s\nNumber of runs: %d\nMean execution time: %.4f seconds\nStandard deviation: %.4f seconds\nIndividual run times: %s\n",
    output_prefix,
    toString(first_prediction),
    n_runs,
    mean_time,
    sd_time,
    toString(round(times, 4))
  )
  
  # Write to file
  write(result, file=paste0(output_prefix, "_results.txt"), append=TRUE)
}

#Shen test
model_path <- "nfv.fasta2mlp_model"
fasta_path <- "seq_test.fasta"
output_prefix <-"shen_mlp"
result <- load_and_predict(model_path, fasta_path, output_prefix)

#In-house test
model_path <- "nfv.fasta5mlp_model"
fasta_path <- "seq_test.fasta"
output_prefix <-"inhouse_mlp"
result <- load_and_predict(model_path, fasta_path, output_prefix)

#Steiner test
model_path <- "nfv_train.fasta2mlp_model"
fasta_path <- "seq_test.fasta"
output_prefix <-"steiner_mlp"
result <- load_and_predict(model_path, fasta_path, output_prefix)