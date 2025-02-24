library("stringr")

HIVDBtreatment <- function(dataset, reference_sequence, sequence_column_range) {
  # Clean reference sequence by removing newlines
  clean_ref_seq <- str_replace_all(reference_sequence, "[\r\n]", "")
  ref_seq_vector <- strsplit(clean_ref_seq, "")[[1]]
  ref_seq_matrix <- t(data.frame(ref_seq_vector))
  
  # Filter dataset for PhenoSense method and Subtype B
  filtered_dataset <- dataset[dataset$Method == "PhenoSense" & dataset$Subtype == "B", ]
  
  # Process sequences
  sequence_columns <- filtered_dataset[, sequence_column_range]
  
  # Replace gaps with reference sequence
  for (col in 1:ncol(sequence_columns)) {
    gap_indices <- which(sequence_columns[, col] == "-")
    sequence_columns[gap_indices, col] <- ref_seq_matrix[1, col]
  }
  
  # Update sequences in the main dataset
  filtered_dataset[, sequence_column_range] <- sequence_columns
  
  # Remove problematic sequences (containing stop codons, deletions, insertions, or ambiguities)
  problem_indices <- which(filtered_dataset[,] == "*" |
                             filtered_dataset[,] == "~" |
                             filtered_dataset[,] == "X" |
                             filtered_dataset[,] == "." |
                             filtered_dataset[,] == "#", arr.ind = TRUE)
  
  if (length(problem_indices[,1]) > 0) {
    filtered_dataset <- filtered_dataset[-c(problem_indices[,1]), ]
  }
  
  # Add reference sequence as first row
  ref_row <- filtered_dataset[1,]
  ref_row[1,1] <- 1
  ref_row[1,sequence_column_range] <- ref_seq_matrix
  filtered_dataset <- rbind(ref_row, filtered_dataset)
  
  return(filtered_dataset)
}

format_sequence_dataset <- function(dataset, sequence_columns) {
  formatted_data <- dataset
  sequences <- apply(dataset[, sequence_columns], 1, paste, collapse="")
  formatted_data <- formatted_data[, !(names(formatted_data) %in% names(dataset)[sequence_columns])]
  formatted_data$Sequence <- sequences
  desired_order <- c("SeqID", "PtID", "Subtype", "Method", "RefID", "Type", 
                     "IsolateName", "SeqType", "FPV", "ATV", "IDV", "LPV", 
                     "NFV", "SQV", "TPV", "DRV", "Sequence")
  formatted_data <- formatted_data[, desired_order]
  return(formatted_data)
}


# Main processing pipeline
setwd('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/')

# Load and process protease dataset
protease_dataset <- read.csv("PI_DataSet.Full.txt", sep = '\t', stringsAsFactors = FALSE)
reference_sequence <- "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

processed_data <- HIVDBtreatment(protease_dataset, reference_sequence, 17:115)
ref_sequence <- processed_data[1, ]
processed_data <- processed_data[-1, ]

# Handle duplicate sequences
duplicate_seqs <- processed_data[duplicated(processed_data[,17:115]) | duplicated(processed_data[,17:115], fromLast = TRUE),]
merged_duplicates <- data.frame()
for(i in 1:nrow(duplicate_seqs)){
  tmp <- duplicate_seqs[which(duplicate_seqs[i,116] == duplicate_seqs[,116]),]
  tmp$FPV <- median(tmp$FPV, na.rm = T)
  tmp$ATV <- median(tmp$ATV, na.rm = T)
  tmp$IDV <- median(tmp$IDV, na.rm = T)
  tmp$LPV <- median(tmp$LPV, na.rm = T)
  tmp$NFV <- median(tmp$NFV, na.rm = T)
  tmp$SQV <- median(tmp$SQV, na.rm = T)
  tmp$TPV <- median(tmp$TPV, na.rm = T)
  tmp$DRV <- median(tmp$DRV, na.rm = T)
  merged_duplicates <- rbind(merged_duplicates, tmp)
}
merged_duplicates <- merged_duplicates[!duplicated(merged_duplicates[,17:115]), ]

# Remove duplicate sequences and combine final dataset
duplicate_ids <- as.vector(duplicate_seqs$SeqID)
final_dataset <- processed_data[!(processed_data$SeqID %in% duplicate_ids), ]
final_dataset <- rbind(final_dataset, merged_duplicates)
final_dataset <- rbind(ref_sequence, final_dataset)

# Process sequences for ambiguity detection
sequence_data <- final_dataset[, -116]
for (i in 17:ncol(sequence_data)) {
  sequence_data[, i] <- as.character(sequence_data[, i])
}

# Find sequences with ambiguous positions
ambiguous_indices <- c()
for (i in 1:nrow(sequence_data)) {
  for (j in 17:115) {
    if (nchar(sequence_data[i, j]) > 1 & !is.na(sequence_data[i, j])) {
      ambiguous_indices <- c(ambiguous_indices, i)
      break
    }
  }
}
ambiguous_indices <- unique(ambiguous_indices)
ambiguous_sequences <- sequence_data[ambiguous_indices, ]

single_sequences <- sequence_data[-ambiguous_indices, ]

# Export ambiguous sequences for further processing
output_filename <- "shen_data/ambiguous_protease_sequences.csv"
write.csv(ambiguous_sequences, output_filename, row.names = FALSE)

# Export all sequences for further processing 
output_filename <- "shen_data/all_protease_sequences.csv"
write.csv(sequence_data, output_filename, row.names = FALSE)

# Export single sequences for further processing 
output_filename <- "shen_data/single_protease_sequences.csv"
write.csv(single_sequences, output_filename, row.names = FALSE)
