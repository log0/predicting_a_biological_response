superman <- makeCluster(7)
registerDoSNOW(superman)
getDoParRegistered(); getDoParName(); getDoParWorkers();

###Run 28 oRF.pls models
oRF.pls.cvs <- foreach(
  i=1:nfolds
  ,.packages='obliqueRF'
  ,.verbose=TRUE
  ) %dopar% {
    #i<-1L
    train.flag <- (fold.ids != i)
    test.flag <- (fold.ids == i)
    
    ###Pass the gbm the out of fold data too to save time
    trash.oRF <- obliqueRF(
      x=as.matrix(train[train.flag,])
      ,y=as.numeric(outcome[train.flag])
      ,mtry=250
      ,ntree=500
      ,training_method="pls"
      )
    oRF.fold.pred <- predict(trash.oRF,train[test.flag,],type="prob")
    return(oRF.fold.pred[,2])
  }

stopCluster(superman)
stop.time <- date()