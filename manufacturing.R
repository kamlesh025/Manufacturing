library(randomForest)
library(ggplot2)
library(dplyr)
library(tree)
library(cvTools)
library(pROC)
library(caret)
 

getwd()
setwd("G:\\Data Science\\Course Project\\Manufacturing Project\\Submission")

mf_train = read.csv("product_train.csv",stringsAsFactors = F)
mf_test = read.csv("product_test.csv",stringsAsFactors = F)
names(mf_train)

mf_test$went_on_backorder = NA

mf_train$data = 'train'
mf_test$data = 'test'

mf_all <- rbind(mf_train,mf_test)

# To find any missing values in complete data

apply(mf_all,2, function(x) sum(is.na(x)))

# No missing values found

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}

library(dplyr)

str(mf_all)

names(mf_all)[sapply(mf_all,function(x) is.character(x))]

apply(mf_all,2,function(x) length(unique(x)))

glimpse(mf_all)

cat_col = c('potential_issue','deck_risk','oe_constraint','ppap_risk','stop_auto_buy','rev_stop')

table(mf_all$potential_issue)
table(mf_all$deck_risk)
table(mf_all$oe_constraint)
table(mf_all$ppap_risk)
table(mf_all$stop_auto_buy)
table(mf_all$rev_stop)


for(cat in cat_col){
  mf_all = CreateDummies(mf_all,cat,50)
}

glimpse(mf_all)
str(mf_all)

lapply(mf_all,function(x) sum(is.na(x)))

for(col in names(mf_all)){
  
  if(sum(is.na(mf_all[,col]))>0 & !(col %in% c("data","went_on_backorder"))){
    
    r_all[is.na(r_all[,col]),col]=mean(r_all[,col],na.rm=T)
  }
}

mf_all$went_on_backorder= as.numeric(mf_all$went_on_backorder=='Yes')

str(mf_all)

library(dplyr)
mf_train= mf_all %>% filter(data=='train') %>% select(-data)
mf_test= mf_all%>% filter(data=='test') %>% select(-data,-went_on_backorder)
 
str(mf_train)

set.seed(3)
r <- sample(1:nrow(mf_train),0.8*nrow(mf_train))
mf_train1 = mf_train[r,]
mf_train2 = mf_train[-r,]

##########################################
library(car)

for_vif=lm(went_on_backorder~.-sku,data=mf_train1)
sort(vif(for_vif),decreasing = T)

for_vif=lm(went_on_backorder~.-sku-forecast_9_month,data=mf_train1)
sort(vif(for_vif),decreasing = T)

for_vif=lm(went_on_backorder~.-sku-forecast_9_month-sales_6_month,data=mf_train1)
sort(vif(for_vif),decreasing = T)

for_vif=lm(went_on_backorder~.-sku-forecast_9_month-sales_6_month-sales_3_month,data=mf_train1)
sort(vif(for_vif),decreasing = T)


for_vif=lm(went_on_backorder~.-sku-forecast_9_month-sales_6_month-sales_3_month-forecast_6_month,data=mf_train1)
sort(vif(for_vif),decreasing = T)

for_vif=lm(went_on_backorder~.-sku-forecast_9_month-sales_6_month-sales_3_month-forecast_6_month-sales_9_month,data=mf_train1)

sort(vif(for_vif),decreasing = T)


## By using random Forest model

library(randomForest)
library(pROC)

rf.tree=randomForest(factor(went_on_backorder)~.-sku, data=mf_train1, do.trace = T,ntree = 200)

## predict the score on testing data
predict.score=predict(rf.tree,newdata=mf_train2,type='prob')[,2]

## obtaining auc on testing_data ---------------------------------------------------------------------------------------------------------------

plot(roc(as.numeric(mf_train2$went_on_backorder),as.numeric(predict.score)))
auc(roc(as.numeric(mf_train2$went_on_backorder),as.numeric(predict.score)))

### Making on final model

final.model =randomForest(factor(went_on_backorder)~.-sku,data=mf_train,do.trace = F,ntree = 200)

## predict the probability on test_data 

final.probability.prediction = predict(final.model,newdata= mf_test,type='prob')[,2]

## To give answer in hard classes

train.score=predict(final.model,newdata = mf_train,type = 'prob')[,2]

real=mf_train$went_on_backorder
cutoffs=seq(0.001,0.999,0.001)
cutoff_data=data.frame(cutoff=99,Sn=99,Sp=99,KS=99,F5=99,F.1=99,M=99)

for(cutoff in cutoffs){
  
  predicted=as.numeric(train.score>cutoff)
  
  TP=sum(real==1 & predicted==1)
  TN=sum(real==0 & predicted==0)
  FP=sum(real==0 & predicted==1)
  FN=sum(real==1 & predicted==0)
  
  P=TP+FN
  N=TN+FP
  
  Sn=TP/P
  Sp=TN/N
  precision=TP/(TP+FP)
  recall=Sn
  
  KS=(TP/P)-(FP/N)
  F5=(26*precision*recall)/((25*precision)+recall)
  F.1=(1.01*precision*recall)/((.01*precision)+recall)
  
  M=(4*FP+FN)/(5*(P+N))
  
  cutoff_data=rbind(cutoff_data,c(cutoff,Sn,Sp,KS,F5,F.1,M))
}

cutoff_data=cutoff_data[-1,]

my_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]

my_cutoff

score = 1-(0.025/max(cutoff_data$KS))

# predicting in form of 1 and 0

final.test.prediction =as.numeric(final.probability.prediction > my_cutoff)

final.test.prediction = as.character(final.test.prediction == 1)
final.test.prediction = gsub("FALSE","No",final.test.prediction)
final.test.prediction = gsub("TRUE","Yes",final.test.prediction)

table(final.test.prediction)

write.csv(final.test.prediction,"Kamalesh_P3_part2.csv",row.names = F)













# To create confusion matrix to calculate KS score by using bank_train2 dataset

#train.score1 = predict(log_fit_final,newdata = mf_train2,type='response')
train.predicted = as.numeric(predict.score> 0.31)
table(train.predicted)

#library(caret)

confusionMatrix(as.factor(train.predicted),as.factor(mf_train2$went_on_backorder))



