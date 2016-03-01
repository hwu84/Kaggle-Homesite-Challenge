
# find feature importance of bst19_1. Select certain proportaion of feature(top 200) and train new xgb
# Author: Haozhen Wu

bst19_1 = xgb.load("bst19_1")
load("homesite19_submit.RData")
homesite19_train
feature_importance <- xgb.importance(feature_names = colnames(homesite19_train), model = bst19_1)
#png("feature_importance_homesite21.png",units = "in", width=11, height=8.5, res=300)
xgb.plot.importance(importance_matrix =feature_importance )
#dev.off()
importantFeatures = feature_importance[[1]][1:150]

# convert factor into numeric id.
# added more features related to date. row sum of 0,-1,0 and -1.
# add interaction feature. dayofweek + 10*SalesField7

setwd("~/Desktop/study/kaggle/Homesite")
library(xgboost)
library(readr)
library(chron)
set.seed(1120)

cat("reading the train and test data\n")
train <- read_csv("~/Desktop/study/kaggle/Homesite/train.csv")
test <- read_csv("~/Desktop/study/kaggle/Homesite/test.csv")

# There are some NAs in the integer columns so conversion to zero

train[is.na(train)]   <- -1 # change from 0 to -1
test[is.na(test)]   <- -1

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)


# seperating out the elements of the date column for the train set
# features related to date: month, year, day of week, day of month, day of year, week of year, since 2000

train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$dayOfWeek <- weekdays(as.Date(train$Original_Quote_Date))
train$dayOfMonth <- as.integer(days(as.Date(train$Original_Quote_Date)))
train$dayOfYear <- as.integer(julian(as.Date(train$Original_Quote_Date),origin = as.Date("2013-01-01")))
for (i in 1:length(train$Original_Quote_Date)){
  if (train$dayOfYear[i]>=365 & train$dayOfYear[i]< 730){
    train$dayOfYear[i] = train$dayOfYear[i] - 365
  }else if(train$dayOfYear[i]>=730){
    train$dayOfYear[i] = train$dayOfYear[i] - 730
  }
}
train$dayOfYear <- as.integer(train$dayOfYear)
train$weekOfYear <- as.integer( format(as.Date(train$Original_Quote_Date)+3, "%U"))
train$daySince130101 <- as.integer(julian(as.Date(train$Original_Quote_Date),origin = as.Date("2013-01-01")))

# removing the date column
train <- train[,-c(2)]
train$dayOfWeek = as.integer(factor(train$dayOfWeek))

# seperating out the elements of the date column for the train set
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$dayOfWeek <- weekdays(as.Date(test$Original_Quote_Date))
test$dayOfMonth <- as.integer(days(as.Date(test$Original_Quote_Date)))
test$dayOfYear <- as.integer(julian(as.Date(test$Original_Quote_Date),origin = as.Date("2013-01-01")))
for (i in 1:length(test$Original_Quote_Date)){
  if (test$dayOfYear[i]>=365 & test$dayOfYear[i]< 730){
    test$dayOfYear[i] = test$dayOfYear[i] - 365
  }else if(test$dayOfYear[i]>=730){
    test$dayOfYear[i] = test$dayOfYear[i] - 730
  }
}
test$dayOfYear <- as.integer(test$dayOfYear)
test$weekOfYear <- as.integer( format(as.Date(test$Original_Quote_Date)+3, "%U"))
test$daySince130101 <- as.integer(julian(as.Date(test$Original_Quote_Date),origin = as.Date("2013-01-01")))

# removing the date column
test <- test[,-c(2)]
test$dayOfWeek = as.integer(factor(test$dayOfWeek))

# row sum of number of -1, 0, -1+0
number_negOne = apply(train[,c(3:298)],1,function(x) length(which(x==-1)))
number_zero = apply(train[,c(3:298)],1,function(x) length(which(x==0)))
number_negOne_zero = apply(train[,c(3:298)],1,function(x) length(which(x==0|x==-1)))
train$number_negOne = number_negOne
train$number_zero = number_zero
train$number_negOne_zero = number_negOne_zero

number_negOne = apply(test[,c(2:297)],1,function(x) length(which(x==-1)))
number_zero = apply(test[,c(2:297)],1,function(x) length(which(x==0)))
number_negOne_zero = apply(test[,c(2:297)],1,function(x) length(which(x==0|x==-1)))
test$number_negOne = number_negOne
test$number_zero = number_zero
test$number_negOne_zero = number_negOne_zero

# new interaction feature from Gert
train$DOW_plus_10SF7 = train$dayOfWeek + 10* as.integer(factor(train$SalesField7))
test$DOW_plus_10SF7 = test$dayOfWeek + 10* as.integer(factor(test$SalesField7))

#
feature.names <- names(train)[c(3:309)]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")

k = 0
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    k = k+1
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

importantFeatures = feature_importance[[1]][1:200]

homesite22_train<-train[,importantFeatures]
homesite22_test = test[,importantFeatures]
test_id = test$QuoteNumber


dtrain_homesite22<-xgb.DMatrix(data=data.matrix(homesite22_train),label=train$QuoteConversion_Flag)
watchlist22<-list(train=dtrain_homesite22)
xgb.DMatrix.save(dtrain_homesite22, "dtrain_homesite22.buffer")
save(list = c("homesite22_test","homesite22_train","test_id"),file="homesite22_submit.RData")
load("homesite22_submit.RData")


# model 22_1
folds5 <- read.csv("folds5.txt")
folds5_list = list()
folds5_list[[1]] = which(folds5$fold==0)
folds5_list[[2]] = which(folds5$fold==1)
folds5_list[[3]] = which(folds5$fold==2)
folds5_list[[4]] = which(folds5$fold==3)
folds5_list[[5]] = which(folds5$fold==4)

sink('bst22_1_5foldcv.txt')  
time_start=proc.time()
param <- list(  objective           = "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.01, # 0.01
                max_depth           = 7, #changed from default of 8
                subsample           = 0.83,  
                colsample_bytree    = 0.5, 
                num_parallel_tree   = 1,
                min_child_weight    = 1,
                gamma               = 12
                # alpha = 0.0001, 
                # lambda = 1
)
set.seed(1120)
bst22_1_5cv = xgb.cv(param, dtrain_homesite22,
                     nrounds = 25000, #30000
                     #nfold=5,
                     folds = folds5_list,
                     metrics={'error'}, 
                     verbose = 1, 
                     print.every.n = 20,
                     maximize = TRUE,
                     nthread = 15,
                     prediction = TRUE
)
time = time_start-proc.time()
time
sink()

bst22_1_5cv_holdout = data.frame(Holdout_Pred = bst22_1_5cv$pred)
write.csv(bst22_1_5cv_holdout, file = "bst22_1_5cv_holdout.csv",row.names = F)



sink('bst22_1_out.txt')  
time_start=proc.time()
param <- list(  objective           = "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.01, # 0.01
                max_depth           = 7, #changed from default of 8
                subsample           = 0.83,  
                colsample_bytree    = 0.5, 
                num_parallel_tree   = 1,
                min_child_weight    = 1,
                gamma               = 12
                # alpha = 0.0001, 
                # lambda = 1
)
set.seed(1120)
bst22_1 <- xgb.train(   params              = param, 
                        data                = dtrain_homesite22, 
                        nrounds             = 25000, 
                        verbose             = 1,  #1
                        early.stop.round    = 10000,
                        watchlist           = watchlist22,
                        maximize            = T,
                        print.every.n = 20,
                        nthread             = 15
)
time = time_start-proc.time()
time
xgb.save(bst22_1, "bst22_1")
sink()

bst22_1 = xgb.load("bst22_1")
load("homesite22_submit.RData")
pred22_1_t2 <- predict(bst22_1, data.matrix(homesite22_test), ntreelimit = 25000)
result22_1_t2<- data.frame(QuoteNumber=test_id, QuoteConversion_Flag=pred22_1_t2)
write.csv(result22_1_t2, file = "result22_1_t2.csv",row.names = F)


