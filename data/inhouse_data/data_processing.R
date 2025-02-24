library("stringr")

HIVDBtreatment <- function(dataset, refseq, sequence_columns) {
  ref_seq <- str_replace_all(refseq, "[\r\n]" , "")
  vetor_refseq <- c(strsplit(ref_seq, ""))
  vetor_refseq <- data.frame(vetor_refseq)
  vetor_refseq <- t(vetor_refseq)
  dataset <- dataset[which(dataset$Method == "PhenoSense" ),] # | dataset$Method == "Antivirogram"
  dataset <- dataset[which(dataset$Subtype == "B"),]
  dataset_temp <- dataset[,sequence_columns]
  
  for (i in 1:ncol(dataset_temp)){
    dataset_temp[which(dataset_temp[,i] == "-"),i] <- vetor_refseq[1,i]
    for (j in 1:nrow(dataset_temp)){
      if (nchar(dataset_temp[j,i]) > 1 & !is.na(dataset_temp[j,i])){
        dataset_temp[j,i] <- "X"
      }
    }
  }
  dataset[,sequence_columns] <- dataset_temp
  testeindexex <- which(dataset[,] == "*" |
                          dataset[,] == "~" |
                          dataset[,] == "X" |
                          dataset[,] == "." |
                          dataset[,] == "#", arr.ind = TRUE)
  if (length(testeindexex[,1]) > 0){
    dataset <- dataset[-c(testeindexex[,1]), ]
  }
  datasetfirst <- dataset[1,]
  datasetfirst[1,1] <- 1
  datasetfirst[1,sequence_columns] <- vetor_refseq
  dataset <- rbind(datasetfirst, dataset)
  return(dataset)
  
}

handle_duplicate_sequences <- function(processed_data, ref_sequence = NULL) {
  duplicate_seqs <- processed_data[duplicated(processed_data[,17:115]) | 
                                     duplicated(processed_data[,17:115], fromLast = TRUE),]
  merged_duplicates <- data.frame()
  for(i in 1:nrow(duplicate_seqs)) {
    tmp <- duplicate_seqs[which(duplicate_seqs[i,116] == duplicate_seqs[,116]),]
    resistance_measures <- c("FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV")
    for(measure in resistance_measures) {
      tmp[[measure]] <- median(tmp[[measure]], na.rm = TRUE)
    }
    merged_duplicates <- rbind(merged_duplicates, tmp)
  }
  merged_duplicates <- merged_duplicates[!duplicated(merged_duplicates[,17:115]), ]
  duplicate_ids <- as.vector(duplicate_seqs$SeqID)
  final_dataset <- processed_data[!(processed_data$SeqID %in% duplicate_ids), ]
  final_dataset <- rbind(final_dataset, merged_duplicates)
  if (!is.null(ref_sequence)) {
    final_dataset <- rbind(ref_sequence, final_dataset)
  }
  
  return(final_dataset)
}

classification_sequence_dataset <- function(dataset, sequence_columns) {
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

multifasta_fromsequence <- function(dataset, sequence_columns, dsname){
  for (i in 1:nrow(dataset)){
    cat(">",dataset[i,1],"\n", unlist(dataset[i,sequence_columns]), "\n", sep = "", append = TRUE, file = paste(dsname,".fasta", sep = ""))
  }
}

# Main processing pipeline
setwd('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/')

# Load and process protease dataset
protease_dataset <- read.csv("PI_DataSet.Full.txt", sep = '\t', stringsAsFactors = FALSE)
reference_sequence <- "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
processed_data <- HIVDBtreatment(protease_dataset,  refseq = reference_sequence, sequence_columns = c(17:115))
ref_sequence <- processed_data[1, ]
processed_data <- processed_data[-1, ]

# Final dataset
final_dataset <- handle_duplicate_sequences(processed_data = processed_data, ref_sequence = ref_sequence)
output_filename <- "inhouse_data/pi_final_dataset.csv"
write.csv(final_dataset, output_filename, row.names = FALSE)

#Classification dataset
pi_sequences_classification <- classification_sequence_dataset(final_dataset, 17:115)
output_filename <- "inhouse_data/pi_sequences_classification.csv"
write.csv(pi_sequences_classification, output_filename, row.names = FALSE)

# Export single sequences for further processing 
single_sequences <- id_sequence_dataset(final_dataset, 17:115)
output_filename <- "inhouse_data/single_protease_sequences.csv"
write.csv(single_sequences, output_filename, row.names = FALSE)

# Write multifasta
multifasta_fromsequence(dataset = final_dataset, sequence_columns = c(17:115), dsname = "inhouse_data/pi_phenosense")
