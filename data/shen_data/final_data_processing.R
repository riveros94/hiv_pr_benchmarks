
clean_problem_chars <- function(df) {
  sequences <- df[, 17:115]
  problem_indices <- which(sequences == "*" |
                             sequences == "~" |
                             sequences == "X" |
                             sequences == "." |
                             sequences == "#",
                           arr.ind = TRUE)
  
  if (length(problem_indices) > 0) {
    rows_to_remove <- unique(problem_indices[,1])
    clean_df <- df[-rows_to_remove, ]
    return(clean_df)
  }
  
  return(df)
}

format_sequence_dataset <- function(dataset, sequence_columns) {
  formatted_data <- dataset
  threshold_rules <- list(
    NFV = 3,
    SQV = 3,
    IDV = 3,
    FPV = 4,
    ATV = 3,
    LPV = 9,
    TPV = 2,
    DRV = 10
  )
  
  transform_drug <- function(x, threshold) {
    ifelse(is.na(x), NA,
           ifelse(x >= threshold, 1, 0))
  }
  
  for (drug in names(threshold_rules)) {
    if (drug %in% names(formatted_data)) {
      formatted_data[[drug]] <- transform_drug(formatted_data[[drug]], 
                                               threshold_rules[[drug]])
    }
  }
  
  sequences <- apply(dataset[, sequence_columns], 1, paste, collapse="")
  formatted_data <- formatted_data[, !(names(formatted_data) %in% names(dataset)[sequence_columns])]
  formatted_data$Sequence <- sequences
  desired_order <- c("SeqID", "PtID", "Subtype", "Method", "RefID", "Type", 
                     "IsolateName", "SeqType", "FPV", "ATV", "IDV", "LPV", 
                     "NFV", "SQV", "TPV", "DRV", "Sequence")
  formatted_data <- formatted_data[, desired_order]
  
  return(formatted_data)
}

id_sequence_dataset <- function(dataset, sequence_columns) {
  sequences <- apply(dataset[, sequence_columns], 1, paste, collapse="")
  
  formatted_data <- data.frame(
    SeqID = dataset$SeqID,
    Sequence = sequences
  )
  
  return(formatted_data)
}

create_drug_fastas <- function(formatted_data) {
  drugs <- c("NFV", "SQV", "IDV", "FPV", "ATV", "LPV", "TPV", "DRV")
  formatted_data <- formatted_data[-1, ]
  for (drug in drugs) {
    valid_sequences <- formatted_data[!is.na(formatted_data[[drug]]), ]
    output_file <- paste0(tolower(drug), ".fasta")
    con <- file(output_file, "w")
    for (i in 1:nrow(valid_sequences)) {
      header <- paste0(">", valid_sequences$SeqID[i], "_", 
                       valid_sequences[[drug]][i])
      writeLines(header, con)
      writeLines(valid_sequences$Sequence[i], con)
    }
    close(con)
    cat(sprintf("Arquivo %s criado com %d sequências\n", 
                output_file, nrow(valid_sequences)))
  }
}

setwd('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/shen_data/')
expanded_data <- read.csv('data_18.csv', header = FALSE, tryLogical = FALSE) # Expanded data
single <- read.csv('single_protease_sequences.csv', tryLogical = FALSE) # Single sequences
colnames(expanded_data) <- colnames(single) 
total <- rbind(single, expanded_data) 
cleaned_total <- clean_problem_chars(total) # Clear X, *, ~, ., #
write.csv(cleaned_total, 'protease_expanded_data.csv')
pi_sequences <- format_sequence_dataset(cleaned_total, sequence_columns = c(17:115))
write.csv(pi_sequences, 'pi_sequences_classification.csv')
pi_id_seq <- id_sequence_dataset(cleaned_total, sequence_columns = c(17:115))
write.csv(pi_id_seq, 'pi_id_sequence.csv')
