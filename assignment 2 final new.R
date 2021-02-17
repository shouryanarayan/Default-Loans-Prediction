
#Assignment 2 - Ashok Bhatraju, Shourya Narayan, Vivek Kumar


install.packages('stringr')
library(stringr)
install.packages("ROSE")
library(ROSE)
install.packages('caret')
library(caret)
install.packages("glmnet")
library(glmnet)
library(randomForest)
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(ROCR)
library(rpart)
library(C50)
library(lubridate)
library(gbm)
library(ranger)
lcData4m <- read.csv("C:/Users/ashok/Desktop/UIC/spring 2020/572/assignment 1/lcData4m.csv")

#Data Leakage
#Separating columns that have NA greater than 60%
NaData <- lcData4m[, colMeans(is.na(lcData4m)) > 0.60]
NonNaData <- lcData4m[, colMeans(is.na(lcData4m)) < 0.60]
'ZeroVar <- names(NonNaData) [nearZeroVar(NonNaData)]
ZeroVar'
#Removing variables that can cause data leakage
NonNaData <- NonNaData %>% select(-c(
  #funded_amnt,
  term,funded_amnt_inv,emp_title,emp_length,home_ownership,
  #issue_d,
  pymnt_plan,title,zip_code,addr_state,earliest_cr_line,inq_last_6mths,out_prncp,out_prncp_inv,
  #total_pymnt,
  total_pymnt_inv,total_rec_prncp,total_rec_int,total_rec_late_fee,recoveries,collection_recovery_fee,
  #last_pymnt_d,
  last_pymnt_amnt,last_credit_pull_d,last_fico_range_high,last_fico_range_low,collections_12_mths_ex_med,policy_code,application_type,bc_util,hardship_flag))
#Removing the new derived attributes and the variables used to calculate them as they will cause data leakage
NonNaData <- NonNaData %>% select(-c(num_bc_sats, num_bc_tl, open_acc, returnRateAnnual, propSatBC, propOpenAcct))
#Are there missing values?
any(is.na(NonNaData))

#proportion of missing values in different variables
per <- function(x) {sum(is.na(x))/length(x)*100}
apply(NonNaData, 2, per)
#Handling missing values
NA0to60 <- NonNaData[, colMeans(is.na(NonNaData)) > 0.00]
apply(NA0to60, 2, per)
#Handling mths_since_last_delinq
NonNaData <- NonNaData %>% mutate(mths_since_last_delinq = replace(mths_since_last_delinq, is.na(mths_since_last_delinq), 500))
#Handling bc_open_to_buy
a <- is.numeric(median(NonNaData$bc_open_to_buy))
NonNaData <- NonNaData %>% mutate(bc_open_to_buy = replace(bc_open_to_buy, is.na(bc_open_to_buy), a))
#Handling revol_util
NonNaData$revol_util <- as.character(NonNaData$revol_util)
NonNaData$revol_util <- as.numeric(substr(NonNaData$revol_util, 1, nchar(NonNaData$revol_util)-1))
b = is.numeric(median(NonNaData$revol_util))
NonNaData <- NonNaData %>% mutate(revol_util = replace(revol_util, is.na(revol_util), b))
#Handling mths_since_recent_bc
c <- is.numeric(median(NonNaData$mths_since_recent_bc))
NonNaData <- NonNaData %>% mutate(mths_since_recent_bc = replace(mths_since_recent_bc, is.na(mths_since_recent_bc), c))
#Handling mths_since_recent_inq
d <- is.numeric(median(NonNaData$mths_since_recent_inq))
NonNaData <- NonNaData %>% mutate(mths_since_recent_inq = replace(mths_since_recent_inq, is.na(mths_since_recent_inq), d))
#Handling num_tl_120dpd_2m
e <- is.numeric(median(NonNaData$num_tl_120dpd_2m))
NonNaData <- NonNaData %>% mutate(num_tl_120dpd_2m = replace(num_tl_120dpd_2m, is.na(num_tl_120dpd_2m), e))
#Handling percent_bc_gt_75
f <- is.numeric(median(NonNaData$percent_bc_gt_75))
NonNaData <- NonNaData %>% mutate(percent_bc_gt_75 = replace(percent_bc_gt_75, is.na(percent_bc_gt_75), f))
#Handling mo_sin_old_il_acct
g <- is.numeric(median(NonNaData$mo_sin_old_il_acct))
NonNaData <- NonNaData %>% mutate(mo_sin_old_il_acct = replace(mo_sin_old_il_acct, is.na(mo_sin_old_il_acct), g))
#Handling NA values in last_pymnt_dt
NonNaData<-NonNaData%>%replace_na(list(last_pymnt_d = "Jan-2018"))
#Checking if there are any NA values left
any(is.na(NonNaData))
#Converting columns having percentage to numeric values
NonNaData$int_rate <- as.numeric(sub("%", "", NonNaData$int_rate))
NonNaData$revol_util <- as.numeric(sub("%", "", NonNaData$revol_util))


#Question 1

#Training and Test Data Set for Ovun sampling
set.seed(123)
rcount <- nrow(NonNaData)
trnIndx <- sample(1:rcount, size = round(0.6*rcount), replace=FALSE)
trainset <- NonNaData[trnIndx, ]
testset <- NonNaData[-trnIndx, ]

trainset %>% group_by(loan_status) %>% count()
#Sampling of data
install.packages("ROSE")
install.packages('caret')

os_lcdftrn <- ovun.sample(loan_status~.,data=as.data.frame(trainset),
                          na.action=na.pass,method="over",p=0.5)$data
os_lcdftest <- ovun.sample(loan_status~.,data=as.data.frame(testset),
                           na.action=na.pass,method="over",p=0.5)$data
us_lcdftrn <- ovun.sample(loan_status~.,data=as.data.frame(trainset),
                          na.action=na.pass,method="under",p=0.5)$data
bs_lcdftrn <- ovun.sample(loan_status~.,data=as.data.frame(trainset),
                          na.action=na.pass,method="both",p=0.5)$data

#Question 1:
#GBM model 1
gbm_model<-gbm(loan_status~., data = trainset, distribution = 'gaussian', 
               n.tree = 200,bag.fraction = 0.8, shrinkage=0.01, interaction.depth=4,cv.folds=5,n.cores = 16)
summary(gbm_model, 
        cBars = 10,
        method = relative.influence,
        las = 2)
sqrt(min(gbm_model$cv.error))

bestlter<-gbm.perf(gbm_model,method = 'cv')
scores_gbm_model<-predict(gbm_model, newdata=trainset, n.tree= bestlter, type="response")
fitted.scores<-ifelse(scores_gbm_model>0.5,1,0)
head(scores_gbm_model)
pred_gbm_model=prediction(scores_gbm_model, trainset$loan_status, 
                          label.ordering = c("Charged Off", "Fully Paid"))
aucPerf_gbm_model <-performance(pred_gbm_model, "tpr", "fpr")
plot(aucPerf_gbm_model)
aucPerf_gbm_model=performance(pred_gbm_model, "auc")
aucPerf_gbm_model@y.values

#GBM model 2

gbm_model1<-gbm(loan_status~., data = os_lcdftrn, distribution = 'gaussian', 
                n.tree = 200,bag.fraction = 0.5, shrinkage=0.001, interaction.depth=6,cv.folds=5,n.cores = 16)
summary(gbm_model1, 
        cBars = 10,
        method = relative.influence,
        las = 2)

sqrt(min(gbm_model1$cv.error))
bestlter<-gbm.perf(gbm_model1,method = 'cv')
scores_gbm_model1<-predict(gbm_model1, newdata=os_lcdftest, n.tree= bestlter, type="response")
fitted.scores<-ifelse(scores_gbm_model1>0.5,1,0)
head(scores_gbm_model1)
pred_gbm_model1=prediction(scores_gbm_model1, os_lcdftest$loan_status, 
                           label.ordering = c("Charged Off", "Fully Paid"))
aucPerf_gbm_model1 <-performance(pred_gbm_model1, "tpr", "fpr")
plot(aucPerf_gbm_model1)
aucPerf_gbm_model1=performance(pred_gbm_model1, "auc")
aucPerf_gbm_model1@y.values

#GBM model 3

gbm_model2<-gbm(loan_status~., data = os_lcdftrn, distribution = 'gaussian', 
                n.tree = 1000,bag.fraction = 1, shrinkage=0.1, interaction.depth=10,cv.folds=5,n.cores = 16)
summary(gbm_model2, 
        cBars = 10,
        method = relative.influence,
        las = 2)

bestlter<-gbm.perf(gbm_model2,method = 'cv')
scores_gbm_model2<-predict(gbm_model2, newdata=os_lcdftest, n.tree= bestlter, type="response")
fitted.scores<-ifelse(scores_gbm_model2>0.5,1,0)
head(scores_gbm_model2)
pred_gbm_model2=prediction(scores_gbm_model2, os_lcdftest$loan_status, 
                           label.ordering = c("Charged Off", "Fully Paid"))
aucPerf_gbm_model2 <-performance(pred_gbm_model2, "tpr", "fpr")
plot(aucPerf_gbm_model2)
aucPerf_gbm_model2=performance(pred_gbm_model2, "auc")
aucPerf_gbm_model2@y.values

#GBM Grid model
paramGrid <- expand.grid(treeDepth = c(2,5), minNodesize = c(10,30), bagFraction = c(.5,.8,1),
                         shrinkage = c(.001,.01,.1), bestTree=0,minRMSE=0)

for(i in 1:nrow(paramGrid)){
  
  gbm_paramTune <- gbm(formula=loan_status~.,data=subset(os_lcdftrn),
                       distribution ='gaussian',n.trees = 500,
                       interaction.depth=paramGrid$treeDepth[i],
                       n.minobsinnode=paramGrid$minNodesize[i],
                       bag.fraction=paramGrid$bagFraction[i],
                       shrinkage=paramGrid$shrinkage[i],
                       train.fraction=0.7,
                       n.cores=NULL)
  
  paramGrid$bestTree[i]<-which.min(gbm_paramTune$valid.error)
  paramGrid$minRMSE[i]<-sqrt(min(gbm_paramTune$valid.error))
}
paramGrid
summary(gbm_paramTune)
write.table(paramGrid,"C:/Users/ashok/paramgrid",sep="\t")
summary(paramGrid)
bestlter<-gbm.perf(gbm_paramTune)
scores_gbm_paramTune<-predict(gbm_paramTune, newdata=os_lcdftest, n.tree= bestlter, type="response")
head(scores_gbm_paramTune)
plot.gbm(gbm_paramTune,1,bestlter)
plot.gbm(gbm_paramTune,2,bestlter)
plot.gbm(gbm_paramTune,3,bestlter)

#individual based on RMSE
gbm.fit.final<-gbm(loan_status~.,
                   data=subset(os_lcdftrn),
                   distribution="gaussian",
                   n.trees=500,
                   interaction.depth=5,
                   shrinkage=0.1,
                   n.minobsinnode=10,
                   bag.fraction=.8,
                   train.fraction=0.7,
                   n.cores=NULL)

par(mar = c(5, 8, 1, 1))
summary(gbm.fit.final, 
        cBars = 10,
        method = relative.influence,
        las = 2)
bestIter<-gbm.perf(gbm.fit.final)
scores_gbm.fit.final<- predict(gbm.fit.final, newdata=os_lcdftest, n.tree= bestIter, type="response")
pred_gbm.fit.final=prediction(scores_gbm.fit.final,os_lcdftest$loan_status,label.ordering = c("Charged Off", "Fully Paid"))
aucPerf_gbm.fit.final <-performance(pred_gbm.fit.final, "tpr", "fpr")
plot(aucPerf_gbm.fit.final)
aucPerf_gbm.fit.final=performance(pred_gbm.fit.final, "auc")
aucPerf_gbm.fit.final@y.values

#Training and Test Data Set to check larger training performance
rcount <- nrow(NonNaData)
trnIndx <- sample(1:rcount, size = round(0.8*rcount), replace=FALSE)
trainset <- NonNaData[trnIndx, ]
testset <- NonNaData[-trnIndx, ]

trainset %>% group_by(loan_status) %>% count()

#GBM model 1a
gbm_modela<-gbm(loan_status~., data = trainset, distribution = 'gaussian', 
                n.tree = 200,bag.fraction = 0.8, shrinkage=0.01, interaction.depth=4,cv.folds=5,n.cores = 16)
summary(gbm_modela, 
        cBars = 10,
        method = relative.influence,
        las = 2)

bestlter<-gbm.perf(gbm_modela,method = 'cv')
scores_gbm_modela<-predict(gbm_modela, newdata=testset, n.tree= bestlter, type="response")
fitted.scores<-ifelse(scores_gbm_modela>0.5,1,0)
head(scores_gbm_modela)
pred_gbm_modela=prediction(scores_gbm_modela, testset$loan_status, 
                           label.ordering = c("Charged Off", "Fully Paid"))
aucPerf_gbm_modela <-performance(pred_gbm_modela, "tpr", "fpr")
plot(aucPerf_gbm_modela)
aucPerf_gbm_modela=performance(pred_gbm_modela, "auc")
aucPerf_gbm_modela@y.values

#GBM model 2a

gbm_model1a<-gbm(loan_status~., data = trainset, distribution = 'gaussian', 
                 n.tree = 500,bag.fraction = 0.5, shrinkage=0.001, interaction.depth=6,cv.folds=5,n.cores = 16)
summary(gbm_model1a, 
        cBars = 10,
        method = relative.influence,
        las = 2)

bestlter<-gbm.perf(gbm_model1a,method = 'cv')
scores_gbm_model1a<-predict(gbm_model1a, newdata=testset, n.tree= bestlter, type="response")
fitted.scores<-ifelse(scores_gbm_model1a>0.5,1,0)
head(scores_gbm_model1a)
pred_gbm_model1a=prediction(scores_gbm_model1a, testset$loan_status, 
                            label.ordering = c("Charged Off", "Fully Paid"))
aucPerf_gbm_model1a <-performance(pred_gbm_model1a, "tpr", "fpr")
plot(aucPerf_gbm_model1a)
aucPerf_gbm_model1a=performance(pred_gbm_model1a, "auc")
aucPerf_gbm_model1a@y.values

#GBM model 3a

gbm_model2a<-gbm(loan_status~., data = trainset, distribution = 'gaussian', 
                 n.tree = 1000,bag.fraction = 1, shrinkage=0.1, interaction.depth=10,cv.folds=5,n.cores = 16)
summary(gbm_model2a, 
        cBars = 10,
        method = relative.influence,
        las = 2)

bestlter<-gbm.perf(gbm_model2a,method = 'cv')
scores_gbm_model2a<-predict(gbm_model2a, newdata=testset, n.tree= bestlter, type="response")
fitted.scores<-ifelse(scores_gbm_model2a>0.5,1,0)
head(scores_gbm_model2a)
pred_gbm_model2a=prediction(scores_gbm_model2a, testset$loan_status, 
                            label.ordering = c("Charged Off", "Fully Paid"))
aucPerf_gbm_model2a <-performance(pred_gbm_model2a, "tpr", "fpr")
plot(aucPerf_gbm_model2a)
aucPerf_gbm_model2a=performance(pred_gbm_model2a, "auc")
aucPerf_gbm_model2a@y.values

#RF from previous assignment

#Training and Test Data Set for Random Forest
rcount <- nrow(NonNaData)
trnIndx <- sample(1:rcount, size = round(0.7*rcount), replace=FALSE)
trainset <- NonNaData[trnIndx, ]
testset <- NonNaData[-trnIndx, ]
#RF Model for trainset
rfModel = randomForest(factor(trainset$loan_status) ~ ., 
                       data=trainset, ntree=50, importance=TRUE )
#Predict on test data set with default Threshold
rf_pred <- predict(rfModel,testset)
#Confusion Matrix for default Threshold
CM_RF<- confusionMatrix(rf_pred, testset$loan_status, positive="Charged Off")
CM_RF
#ROC and AUC for RF
score=predict(rfModel,testset, type="prob")[,"Charged Off"]
predTest=prediction(score, testset$loan_status, label.ordering = c("Fully Paid", "Charged Off"))
curve <-performance(predTest, "tpr", "fpr")
plot(curve)
abline(a=0, b= 1)
curve=performance(predTest, "auc")
curve@y.values
#Prediction for threshold 0.2
probsTest <- predict(rfModel,testset, type='prob')
threshold <- 0.2
pred      <- factor( ifelse(probsTest[, "Charged Off"] > threshold, 'Charged Off', 'Fully Paid') )
pred      <- relevel(pred, "Charged Off")
#Confusion matrix for threshold=0.2  
confusionMatrix(pred, testset$loan_status)
#ROC and AUC for RF
score=predict(rfModel,testset, type="prob")[,"Charged Off"]
predTest=prediction(score, testset$loan_status, label.ordering = c("Fully Paid", "Charged Off"))
curve <-performance(predTest, "tpr", "fpr")
plot(curve)
abline(a=0, b= 1)
curve=performance(predTest, "auc")
curve@y.values
#Lift Curve for RF
Curve1 <-performance(predTest, "lift", "rpp")
plot(Curve1)

#RF Model
rfModel_2= randomForest(loan_status ~ ., lcdfTrain, ntree=200, importance=TRUE )

#GLM model:

view(os_lcdftrn)

#glmnet without alpha
xD<-os_lcdftrn%>% select(-loan_status)

glm_cvM1<-cv.glmnet(data.matrix(xD), os_lcdftrn$loan_status,
                    family="binomial", type.measure = "class")

plot(glm_cvM1)

scores_glm_cvM1<-predict(glm_cvM1,
                         data.matrix(os_lcdftst%>% select(-loan_status)),
                         s = glm_cvM1$lambda.1se, type="class")
perf <- performance(prediction(scores_glm_cvM1, os_lcdftst$loan_status),"tpr","fpr")
plot( perf )

#glmnet for alpha = 1
xD<-os_lcdftrn%>% select(-loan_status)

glm_cvM2<-cv.glmnet(data.matrix(xD), alpha=1, os_lcdftrn$loan_status,
                    family="binomial", type.measure = "class")

plot(glm_cvM2)
scores_glm_cvM2<-predict(glm_cvM2,
                         data.matrix(os_lcdftst%>% select(-loan_status)),
                         s = glm_cvM2$lambda.1se, type="class")
perf <- performance(prediction(scores_glm_cvM2, os_lcdftst$loan_status),"tpr","fpr")
plot( perf )

#glmnet for alpha = 0
glm_cvM3<-cv.glmnet(data.matrix(xD), alpha=0, os_lcdftrn$loan_status,
                    family="binomial", type.measure = "class")

plot(glm_cvM3)
plot(glm_cvM3)
scores_glm_cvM3<-predict(glm_cvM3,
                         data.matrix(os_lcdftst%>% select(-loan_status)),
                         s = glm_cvM3$lambda.1se, type="class")
perf <- performance(prediction(scores_glm_cvM3, os_lcdftst$loan_status),"tpr","fpr")
plot( perf )


#Question 2

#Training and Test Data Set for Ovun sampling
rcount <- nrow(NonNaData)
trnIndx <- sample(1:rcount, size = round(0.6*rcount), replace=FALSE)
trainset <- NonNaData[trnIndx, ]
testset <- NonNaData[-trnIndx, ]

#Sampling of data
os_lcdftrn <- ovun.sample(loan_status~.,data=as.data.frame(trainset),
                          na.action=na.pass,method="over",p=0.5)$data

#Random Forest for Actual Return
rfModel_Ret <- ranger(actualReturn ~., data=subset(os_lcdftrn, select=-c(annRet_percent, actualTerm, loan_status)), num.trees =200,
                      importance='permutation')
rfPredRet_trn<- predict(rfModel_Ret, os_lcdftrn)
sqrt(mean( (rfPredRet_trn$predictions - os_lcdftrn$actualReturn)^2))
sqrt(mean( ( (predict(rfModel_Ret, testset))$predictions - testset$actualReturn)^2))
plot ( (predict(rfModel_Ret, testset))$predictions, testset$actualReturn)
plot ( (predict(rfModel_Ret, os_lcdftrn))$predictions, os_lcdftrn$actualReturn)

#Performance by deciles for Random Forest
predRet_Trn<-os_lcdftrn%>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet=(predict(rfModel_Ret, os_lcdftrn))$predictions)
predRet_Trn<-predRet_Trn%>% mutate(tile=ntile(-predRet, 10))
predRet_Trn%>% group_by(tile) %>%  summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"), avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"), totB=sum(grade=="B"), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"), totG=sum(grade=="G") )

#Training and Test Data Set for Ovun sampling
rcount <- nrow(NonNaData)
trnIndx <- sample(1:rcount, size = round(0.7*rcount), replace=FALSE)
trainset <- NonNaData[trnIndx, ]
testset <- NonNaData[-trnIndx, ]

#Sampling of data
os_lcdftrn <- ovun.sample(loan_status~.,data=as.data.frame(trainset),
                          na.action=na.pass,method="over",p=0.5)$data

#Random Forest for Actual Return
rfModel_Ret <- ranger(actualReturn ~., data=subset(os_lcdftrn, select=-c(annRet_percent, actualTerm, loan_status)), num.trees =200,
                      importance='permutation')
rfPredRet_trn<- predict(rfModel_Ret, os_lcdftrn)
sqrt(mean( (rfPredRet_trn$predictions - os_lcdftrn$actualReturn)^2))
sqrt(mean( ( (predict(rfModel_Ret, testset))$predictions - testset$actualReturn)^2))
plot ( (predict(rfModel_Ret, testset))$predictions, testset$actualReturn)
plot ( (predict(rfModel_Ret, os_lcdftrn))$predictions, os_lcdftrn$actualReturn)

#Training and Test Data Set for Ovun sampling
rcount <- nrow(NonNaData)
trnIndx <- sample(1:rcount, size = round(0.8*rcount), replace=FALSE)
trainset <- NonNaData[trnIndx, ]
testset <- NonNaData[-trnIndx, ]

#Sampling of data
os_lcdftrn <- ovun.sample(loan_status~.,data=as.data.frame(trainset),
                          na.action=na.pass,method="over",p=0.5)$data

#Random Forest for Actual Return
rfModel_Ret <- ranger(actualReturn ~., data=subset(os_lcdftrn, select=-c(annRet_percent, actualTerm, loan_status)), num.trees =200,
                      importance='permutation')
rfPredRet_trn<- predict(rfModel_Ret, os_lcdftrn)
sqrt(mean( (rfPredRet_trn$predictions - os_lcdftrn$actualReturn)^2))
sqrt(mean( ( (predict(rfModel_Ret, testset))$predictions - testset$actualReturn)^2))
plot ( (predict(rfModel_Ret, testset))$predictions, testset$actualReturn)
plot ( (predict(rfModel_Ret, os_lcdftrn))$predictions, os_lcdftrn$actualReturn)

#GLM for Actual Return (Vanilla)
xD<-os_lcdftrn %>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)

glmRet_cv<- cv.glmnet(data.matrix(xD),os_lcdftrn$actualReturn, family="gaussian")

predRet_Trn <- os_lcdftrn %>% select(grade, loan_status, actualReturn, actualTerm, int_rate)%>%mutate(predRet= predict(glmRet_cv,
                                                                                                                       data.matrix(os_lcdftrn %>% select(-loan_status,-actualTerm, -annRet_percent, -actualReturn)), s="lambda.min"))
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
print(glmRet_cv)


predRet=  predict(glmRet_cv, data.matrix(os_lcdftrn%>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)), s=glmRet_cv$lambda.min )

#RMSE on traning data
sqrt(mean((os_lcdftrn$actualReturn-predRet)^2))

#applying model on test data
predRet_test<-predict(glmRet_cv, data.matrix(testset%>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)), s=glmRet_cv$lambda.min )

#RMSE on test
sqrt(mean((testset$actualReturn-predRet_test)^2))

#GLM for Actual Return (Ridge)
xD<-os_lcdftrn %>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)

glmRet_cv<- cv.glmnet(data.matrix(xD),os_lcdftrn$actualReturn, family="gaussian", alpha=0)

predRet_Trn <- os_lcdftrn %>% select(grade, loan_status, actualReturn, actualTerm, int_rate)%>%mutate(predRet= predict(glmRet_cv,
                                                                                                                       data.matrix(os_lcdftrn %>% select(-loan_status,-actualTerm, -annRet_percent, -actualReturn)), s="lambda.min"))
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
print(glmRet_cv)

#aplying model on traning data
predRet=  predict(glmRet_cv, data.matrix(os_lcdftrn%>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)), s=glmRet_cv$lambda.min, alpha=0)

#RMSE on traning data
sqrt(mean((os_lcdftrn$actualReturn-predRet)^2))

#applying model on test data
predRet_test<-predict(glmRet_cv, data.matrix(testset%>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)), s=glmRet_cv$lambda.min, alpha=0)

#RMSE on test
sqrt(mean((testset$actualReturn-predRet_test)^2))

os_lcdftrn$last_pymnt_d<-as.factor(os_lcdftrn$last_pymnt_d)


#GLM for Actual Return (Ridge)
xD<-os_lcdftrn %>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)

glmRet_cv<- cv.glmnet(data.matrix(xD),os_lcdftrn$actualReturn, family="gaussian", alpha=1)

predRet_Trn <- os_lcdftrn %>% select(grade, loan_status, actualReturn, actualTerm, int_rate)%>%mutate(predRet= predict(glmRet_cv,
                                                                                                                       data.matrix(os_lcdftrn %>% select(-loan_status,-actualTerm, -annRet_percent, -actualReturn)), s="lambda.min"))
predRet_Trn <- predRet_Trn %>% mutate(tile=ntile(-predRet, 10))
predRet_Trn %>% group_by(tile) %>% summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"),
                                             avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totA=sum(grade=="A"),
                                             totB=sum(grade=="B" ), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F") )
print(glmRet_cv)

#aplying model on traning data
predRet=  predict(glmRet_cv, data.matrix(os_lcdftrn%>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)), s=glmRet_cv$lambda.min, alpha=0)

#RMSE on traning data
sqrt(mean((os_lcdftrn$actualReturn-predRet)^2))

#applying model on test data
predRet_test<-predict(glmRet_cv, data.matrix(testset%>% select(-loan_status, -actualTerm, -annRet_percent, -actualReturn)), s=glmRet_cv$lambda.min, alpha=0)

#RMSE on test
sqrt(mean((testset$actualReturn-predRet_test)^2))

plot(glmRet_cv)

#GBM Model for Actual Return
os_lcdftrn$last_pymnt_d<-as.numeric(os_lcdftrn$last_pymnt_d)

gbmModel_Ret <- gbm(formula=actualReturn~., data=subset(os_lcdftrn, select=-c(annRet_percent, actualTerm, loan_status)),
                    distribution = 'gaussian', n.trees = 200, interaction.depth = 5, bag.fraction = 0.5, cv.folds = 5 )

print(gbmModel_Ret)

gbm_PredRet_trn<-predict.gbm(gbmModel_Ret, os_lcdftrn, type="response")


#error on training data (RMSE)
sqrt(mean((gbm_PredRet_trn-os_lcdftrn$actualReturn)^2))

#error on testing data (RMSE)
sqrt(mean((( predict(gbmModel_Ret, testset, type="response"))-testset$actualReturn)^2))

plot((predict(gbmModel_Ret,testset)),testset$actualReturn)
plot((predict(gbmModel_Ret, os_lcdftrn)), os_lcdftrn$actualReturn)

#Question 3

#Question 4
# Drop rows for grade A's and B's:
lg_NonNaData<-subset(NonNaData, grade!="A")
lg_NonNaData<-subset(lg_NonNaData, grade!="B")

#Training and Test Data Set for Ovun sampling
rcount <- nrow(lg_NonNaData)
lg_trnIndx <- sample(1:rcount, size = round(0.6*rcount), replace=FALSE)
lg_trainset <- NonNaData[trnIndx, ]

lg_testset <- NonNaData[-trnIndx, ]

#Sampling of data
lg_os_lcdftrn <- ovun.sample(loan_status~.,data=as.data.frame(lg_trainset),
                             na.action=na.pass,method="over",p=0.5)$data

#Random Forest for Actual Return for lower grade loans
rfModel_Ret <- ranger(actualReturn ~., data=subset(lg_os_lcdftrn, select=-c(annRet_percent, actualTerm, loan_status)), num.trees =200,
                      importance='permutation')
rfPredRet_trn<- predict(rfModel_Ret, lg_os_lcdftrn)
sqrt(mean( (rfPredRet_trn$predictions - lg_os_lcdftrn$actualReturn)^2))
sqrt(mean( ( (predict(rfModel_Ret, lg_testset))$predictions - lg_testset$actualReturn)^2))
plot ( (predict(rfModel_Ret, lg_testset))$predictions, lg_testset$actualReturn)
plot ( (predict(rfModel_Ret, lg_os_lcdftrn))$predictions, lg_os_lcdftrn$actualReturn)

#Performance by deciles for Random Forest of Lower Grade Loans
predRet_Trn<-lg_os_lcdftrn%>% select(grade, loan_status, actualReturn, actualTerm, int_rate) %>% mutate(predRet=(predict(rfModel_Ret, lg_os_lcdftrn))$predictions)
predRet_Trn<-predRet_Trn%>% mutate(tile=ntile(-predRet, 10))
predRet_Trn%>% group_by(tile) %>%  summarise(count=n(), avgpredRet=mean(predRet), numDefaults=sum(loan_status=="Charged Off"), avgActRet=mean(actualReturn), minRet=min(actualReturn), maxRet=max(actualReturn), avgTer=mean(actualTerm), totC=sum(grade=="C"), totD=sum(grade=="D"), totE=sum(grade=="E"), totF=sum(grade=="F"), totG=sum(grade=="G") )

#GBM Model of Actual Return for lower grade loan
lg_os_lcdftrn$last_pymnt_d<-as.numeric(lg_os_lcdftrn$last_pymnt_d)

gbmModel_Ret <- gbm(formula=actualReturn~., data=subset(lg_os_lcdftrn, select=-c(annRet_percent, actualTerm, loan_status)),
                    distribution = 'gaussian', n.trees = 200, interaction.depth = 5, bag.fraction = 0.5, cv.folds = 5)

print(gbmModel_Ret)

gbm_PredRet_trn<-predict.gbm(gbmModel_Ret, lg_os_lcdftrn, type="response")


#error on training data (RMSE)
sqrt(mean((gbm_PredRet_trn-lg_os_lcdftrn$actualReturn)^2))

#error on testing data (RMSE)
sqrt(mean((( predict(gbmModel_Ret, lg_testset, type="response"))-testset$actualReturn)^2))

plot((predict(gbmModel_Ret,lg_testset)),lg_testset$actualReturn)
plot((predict(gbmModel_Ret, lg_os_lcdftrn)), lg_os_lcdftrn$actualReturn)

