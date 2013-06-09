library(nnet);library(doSNOW);library(lattice)
##Load your data here....
##error_only needs to be integer vector of 0/1 with NAs for the test outcomes
##train_flag is a logical vector with FALSE where error_only is NA and TRUE elsewhere

##A simple logloss function, with support for missings and no support for cappings
##Input must be on the logit scale
logloss.func <- function(x) -1*mean(log(plogis(ifelse(error_only==1,x,-1*x))),na.rm=TRUE)

###Put together the dataset of out of fold and final predictions for each base learner
nnet.data <- data.frame(
  gbm=fleshed.gbm
  ,rf=fleshed.rf
  ,oRF.pls=fleshed.oRF.pls
  ,oRF.ridge=fleshed.oRF.ridge
  #,RRF=fleshed.RRF
  ,gbm.gaus=fleshed.gbm.gaus
  ,gbm.ada=fleshed.gbm.ada
  ,PC1=manifold.scaled[,1]
  ,PC2=manifold.scaled[,2]
  ,PC3=manifold.scaled[,3]
  )

superman <- makeCluster(7)
registerDoSNOW(superman)
getDoParRegistered(); getDoParName(); getDoParWorkers();

bagged.nnet <- foreach(
  i=1:582
  ,.packages='nnet'
  ,.verbose=TRUE
  ) %dopar% {
  ##Bag it
  curr.train <- sample.int(sum(train_flag),sum(train_flag),TRUE)
  curr.oob <- c(!(seq(1,sum(train_flag)) %in% curr.train),rep(FALSE,sum(!train_flag)))
  ##Ortho it
  prcomp.fit <- prcomp(nnet.data[curr.train,])
  curr.ortho <- predict(prcomp.fit,nnet.data)
  ##Randomize it
  #hist(rlnorm(1000,log(2),0.75))
  curr.decay <- rlnorm(1,log(2),0.75)
  ##Train it
  nnet.fit <- nnet(
    y=error_only[curr.train]
    ,x=curr.ortho[curr.train,]
    ,entropy=TRUE
    ,size=42,decay=curr.decay
    ,maxit=1420,MaxNWts=4200
    )
  ##Pred it
  oob.pred <- rep(NA,sum(train_flag))
  oob.pred[curr.oob] <- predict(nnet.fit,curr.ortho[curr.oob,])
  return(list(
    decay=curr.decay
    ,oob.pred=oob.pred
    ,test.pred=predict(nnet.fit,curr.ortho[!train_flag,])
    ))
  }

stopCluster(superman)


##Extract the pieces
res.decay <- sapply(bagged.nnet,function(x) x$decay)
res.oob.pred <- sapply(bagged.nnet,function(x) qlogis(x$oob.pred))
res.list.oob.pred <- lapply(bagged.nnet,function(x) qlogis(x$oob.pred))
res.test.pred <- sapply(bagged.nnet,function(x) qlogis(x$test.pred))


##Show extremes
summary(apply(res.oob.pred,2,max,na.rm=TRUE));summary(apply(res.oob.pred,2,min,na.rm=TRUE))


##A single oob prediction for all
logloss.func(apply(res.oob.pred,1,mean,na.rm=TRUE))

##Logloss for each oob learner
res.oob.logloss <- apply(res.oob.pred,2,logloss.func)

##Take a look at the individual tunings
plot(x=res.decay,y=res.oob.logloss,log='x')

##Make a few buckets for the decay values
curr.decay.buckets <- cut(res.decay,breaks=quantile(res.decay,seq(0,1,length.out=20)),include.lowest=TRUE)

##Show the buckets from an ensemble perspective
opt.decay.bucket <- tapply(
  ##List of oobs
  res.list.oob.pred
  ##Buckets of decay values
  ,list(curr.decay.buckets)
  ,function(x) logloss.func(
    apply(
      do.call(cbind,x)
      ,1
      ,mean
      ,na.rm=TRUE
      )
    )
  )
opt.decay.bucket
dotchart(opt.decay.bucket,main='Bagged NNet Performance by Decay Bucket',xlab='OOB Log Loss')

##Show the buckets from an individual basis
opt.decay.bucket.indiv <- tapply(
  ##List of oobs
  res.list.oob.pred
  ##Buckets of decay values
  ,list(curr.decay.buckets)
  ,function(x) mean(sapply(x,logloss.func),na.rm=TRUE,trim=0.1)
  )
opt.decay.bucket.indiv
dotchart(opt.decay.bucket.indiv,main='Individual NNet Performance by Decay Bucket',xlab='OOB Log Loss')


###Overlay the ensemble and individual
overlay.frame <- data.frame(
  decay.range=factor(levels(curr.decay.buckets),levels=levels(curr.decay.buckets))
  ,ensemble=opt.decay.bucket
  ,indiv=opt.decay.bucket.indiv
  )
dotplot(decay.range~ensemble+indiv,data=overlay.frame,auto.key=TRUE,type='b',xlab='OOB Log-Loss',main='Tuning NNet Decay for Stacking')
  

###Make the submission set
submit <- data.frame(
  pred = plogis(apply(res.test.pred,1,mean))
  )
summary(submit)
write.table(
  submit
  ,file=paste(save.dir,'stack_w_bag.nnet.csv',sep='')
  ,col.names=FALSE,row.names=FALSE
  )
