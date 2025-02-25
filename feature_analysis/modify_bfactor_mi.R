library(bio3d)

process_bfactor <- function(file_path) {
  content <- readLines(file_path)
  content <- gsub(",", "", content)
  values <- as.numeric(unlist(strsplit(content, " ")))
  new_bfactor_monomer <- values
  return(new_bfactor_monomer)
}

update_bfactor <- function(pdb, new_bfactor_monomer, modified_pdb) {
  pdbtemp <- read.pdb(pdb)
  pdb_atom <- pdbtemp$atom
  new_bfactor <- rep(new_bfactor_monomer, times = 2)
  pdb_atom[which(pdb_atom$elety == "CA"), 13] <- new_bfactor
  pdbtemp$atom <- pdb_atom
  write.pdb(pdb = pdbtemp, file = modified_pdb)
}

pdb <- '../data/3oxc_edited.pdb'

# Define datasets and drugs
datasets <- c("inhouse", "steiner", "shen")
drugs <- c("nfv", "fpv", "atv", "idv", "lpv", "sqv", "tpv", "drv")

dir.create("pdb_rosetta", showWarnings = FALSE)
dir.create("pdb_zscales", showWarnings = FALSE)

# Rosetta energy terms analysis
for (dataset in datasets) {
  output_dir <- paste0("pdb_rosetta/", dataset)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  for (drug in drugs) {
    mi_file_path <- paste0("rosetta/", dataset, "_mi_score_a_", drug, ".txt")
    output_pdb_path <- paste0(output_dir, "/", dataset, "_3oxc_", drug, ".pdb")
    if (file.exists(mi_file_path)) {
      new_bfactor_monomer <- process_bfactor(mi_file_path)
      update_bfactor(pdb, new_bfactor_monomer, output_pdb_path)
      cat("Processed", dataset, "dataset for", toupper(drug), "\n")
    } else {
      cat("Warning: Input file not found:", mi_file_path, "\n")
    }
  }
}

# zScales analysis
for (dataset in datasets) {
  output_dir <- paste0("pdb_zscales/", dataset)
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  for (drug in drugs) {
    mi_file_path <- paste0("zscales/", dataset, "_mi_score_a_", drug, ".txt")
    output_pdb_path <- paste0(output_dir, "/", dataset, "_3oxc_", drug, ".pdb")
    if (file.exists(mi_file_path)) {
      new_bfactor_monomer <- process_bfactor(mi_file_path)
      update_bfactor(pdb, new_bfactor_monomer, output_pdb_path)
      cat("Processed zScales for", dataset, "dataset and", toupper(drug), "\n")
    } else {
      cat("Warning: Input file not found:", mi_file_path, "\n")
    }
  }
}