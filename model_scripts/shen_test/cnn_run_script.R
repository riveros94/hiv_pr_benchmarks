### CNN Run Script ###
# setwd('/hiv_pr_benchmarks/data/shen_data/nn_models') # train data

#required packages
require(keras)
require(caret)
require(iml)
require(plyr)
set.seed(1234)

#functions

### perf_measure ###
perf_measure<-function(y_ref,y_pred){
  TP=0;
  FP=0;
  TN=0;
  FN=0;
  #positive = resistant (1)
  #negative = not resistant (0)
  for(i in 1:length(y_pred)){
    if(y_ref[i]==1&&y_pred[i]==1)
      TP<-TP+1;
    if(y_ref[i]==0&&y_pred[i]==0)
      TN<-TN+1;
    if(y_ref[i]==0&&y_pred[i]==1)
      FP<-FP+1;
    if(y_ref[i]==1&&y_pred[i]==0)
      FN<-FN+1;
  }
  x<-c(TP,FP,TN,FN)
  return(x)
}

### get_model() ###
get_model<-function(bp,nunits){
  #create model
  model<-keras_model_sequential() %>%
    layer_embedding(input_dim=bp,output_dim=128,input_length = bp) %>%
    layer_conv_1d(filters=32,kernel_size=9,activation="relu") %>%
    layer_max_pooling_1d(pool_size = 5)%>%
    layer_conv_1d(filters=32,kernel_size=9,activation="relu") %>%
    layer_flatten() %>%
    layer_dense(units=1,activation="sigmoid")

  #compile model
  model %>% compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=c("accuracy")
  )
}


input_list = c("atv_train.fasta", "drv_train.fasta", "fpv_train.fasta", "idv_train.fasta", "lpv_train.fasta", "nfv_train.fasta", "sqv_train.fasta", "tpv_train.fasta")

#iterate through input files
for (i in 1:length(input_list)){

  #input/output filenames
  infile=input_list[i]
  #pdfout="etr_featureimp_plot.pdf"
  if (i %in% c(1,2,3,4,5,6,7,8)) genecode = "PR"
  else genecode="RT"
  outfile=paste(infile,".cnn.output.txt",sep="")

  #begin writing to outfile
  sink(outfile)

  ### data input and formatting ###

  if(genecode=="RT"){
    bp<-240
    nunits<-99
  }
  if(genecode=="PR"){
    bp<-99
    nunits<-33
  }
  fasta<-read.table(infile,stringsAsFactors = FALSE,comment.char = "")
  data<-array(dim=c(nrow(fasta),2))
  for(i in 1:nrow(fasta)){
    if(i%%2==0){ #even (sequences)
      data[i-1,2]<-fasta[i,]
    }
    else #odd
    {
      data[i,1]<-fasta[i,]
    }
  }#end for
  evens<-seq(0,nrow(fasta),by=2)
  data<-data[-evens,]
  for(i in 1:nrow(data)){ #strip IDs
    data[i,1]<-substr(data[i,1],nchar(data[i,1]),nchar(data[i,1]))
  }#end for

  data_labels<-data[,1]
  data_seqs<-data[,2]

  data_labels<-as.numeric(data_labels)

  #convert characters to integers
  seqs_num<-array(dim=c(length(data_seqs),bp))
  for(i in 1:length(data_seqs)){
    z<-data_seqs[i]
    seq<-unlist(strsplit(z,""))
    for(k in 1:length(seq)){
      seq[k]<-switch(seq[k],
                     "A"=1,
                     "a"=1,
                     "B"=2,
                     "b"=2,
                     "C"=3,
                     "c"=3,
                     "D"=4,
                     "d"=4,
                     "E"=5,
                     "e"=5,
                     "F"=6,
                     "f"=6,
                     "G"=7,
                     "g"=7,
                     "H"=8,
                     "h"=8,
                     "I"=9,
                     "i"=9,
                     "J"=10,
                     "j"=10,
                     "K"=11,
                     "k"=11,
                     "L"=12,
                     "l"=12,
                     "M"=13,
                     "m"=13,
                     "N"=14,
                     "n"=14,
                     "O"=15,
                     "o"=15,
                     "P"=16,
                     "p"=16,
                     "Q"=17,
                     "q"=17,
                     "R"=18,
                     "r"=18,
                     "S"=19,
                     "s"=19,
                     "T"=20,
                     "t"=20,
                     "U"=21,
                     "u"=21,
                     "V"=22,
                     "v"=22,
                     "W"=23,
                     "w"=23,
                     "X"=24,
                     "x"=24,
                     "Y"=25,
                     "y"=25,
                     "Z"=26,
                     "z"=26,
                     "."=27,
                     "#"=28,
                     "~"=29,
                     "*"=30,
                     0
      )
    }
    seqs_num[i,]<-as.integer(seq)
  }

  #convert data into list
  data_list<-c()
  for(i in 1:nrow(seqs_num)){
    seqi<-seqs_num[i,]
    data_list[[i]]<-seqi
  }

  #pad sequences
  data_f <- pad_sequences(
    data_list,
    padding = "post",
    maxlen = bp
  )

  #iterate through 5 folds & save output
  imp_list<-list()
  rocout_list<-list()
  #set number of folds
  k<-5
  #indices<-sample(1:nrow(data_f))
  #folds<-cut(indices,breaks=k,labels=FALSE)
  folds <- createFolds(y = data_labels, k = 5, list = F)
  validation_scores<-c()
  #k-fold cross validation
  starttime=Sys.time()
  for(i in 1:k){
    #separate validation and training data
    validation_data_k<-data_f[which(folds==i),]
    validation_labels_k<-data_labels[which(folds==i)]
    training_data<-data_f[which(folds!=i),]
    training_labels<-data_labels[which(folds!=i)]
    val<-c(validation_data_k,validation_labels_k)

    #class weights
    zero=length(which(training_labels==0))
    one=length(which(training_labels==1))
    weight_0 = 1
    weight_1 = zero/one
    model<-get_model(bp,nunits)

    #train model
    model %>% fit(
      x=training_data,
      y=training_labels,
      epochs=500,
      batch_size=64,
      class_weight=list("0"=weight_0,"1"=weight_1)
    )
    save_model_tf(model, paste(infile,i,"cnn_model",sep=""))
    validation_preds <- model %>% predict(validation_data_k) %>% `>`(0.5) %>% k_cast("int32")
    validation_preds <- as.numeric(validation_preds)
    z<-perf_measure(validation_labels_k,validation_preds)
    print("TP, FP, TN, FN:")
    print(z)


    #ROC
    preds_proba<- predict(model, validation_data_k)
    rocout<-cbind(preds_proba,validation_preds,validation_labels_k)
    rocout_list[[i]] = rocout
  }#end of iteration through folds

  print(validation_scores)
  endtime = Sys.time()
  print("Runtime:")
  print(endtime-starttime)

  # Save results from rocout_list to csv files

  write.csv(rocout_list[[1]],file=paste(infile,"roc.1.cnn.csv",sep=""))
  write.csv(rocout_list[[2]],file=paste(infile,"roc.2.cnn.csv",sep=""))
  write.csv(rocout_list[[3]],file=paste(infile,"roc.3.cnn.csv",sep=""))
  write.csv(rocout_list[[4]],file=paste(infile,"roc.4.cnn.csv",sep=""))
  write.csv(rocout_list[[5]],file=paste(infile,"roc.5.cnn.csv",sep=""))
}
