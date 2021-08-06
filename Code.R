# -----------------------------------------------------------------
# Jovan Trajceski
# Topic: HR Analytics
# -----------------------------------------------------------------

library(ROSE)
library(tidyverse) 
library(modelr) 
library(broom)
library(caret)

data = read.csv('HR_comma_sep.csv')
attach(data)

# Overview of summary (Turnover V.S. Non-turnover)
# cor_vars<-data_set[,c("satisfaction_level","last_evaluation","number_project","average_montly_hours","exp_in_company","Work_accident","left","promotion_last_5years")]
# 
# aggregate(cor_vars[,c("satisfaction_level","last_evaluation","number_project","average_montly_hours","exp_in_company","Work_accident","promotion_last_5years")], by=list(Category=cor_vars$left), FUN=mean)

# Looks like about 76% of employees stayed and 24% of employees left. 
# NOTE: When performing cross validation, its important to maintain this turnover ratio
# EDA on original dataset
attrition<-as.factor(data$left)
summary(attrition)

perc_attrition_rate<-sum(data$left/length(data$left))*100

#percentage of attrition
print(perc_attrition_rate)

data2 = data
#Creating dummy variables
dummy <- dummyVars(" ~ .", data = data2, fullRank = TRUE)
data2 <- data.frame(predict(dummy, newdata = data2))
str(data2)

#####   Build the model with original dataset   #####

## Split the dataset into training and testing
a1 = seq(1,nrow(data2),1)

ind = sample(a1, floor(nrow(data2)*0.8), replace = FALSE) # randomly select 80% of total records from 
# original dataset without replacement
train1 = data2[ind,]
test1 = data2[-ind,]


#####   Estimation   ######
m1 = glm(left~., data=train1, family = binomial(link='logit'))
summary(m1)

###  Prediction / Evaluation   ###
(m1.logodd = predict(m1, newdata=test1[,-7], type="link"))
y_pred = ifelse(m1.logodd>0.5,1,0)
(m1.class = as.numeric(m1.logodd>0)) # Convert z-value into class (0,1)
(m1.right = sum(as.numeric(test1$left == m1.class)))
(m1.acc = mean(as.numeric(test1$left == m1.class)))

### Confusion martix
cm1 = table(test1[, 7], y_pred)
cm1
#m1.acc = (cm1[1,1] + cm1[2,2]) / (cm1[1,1] + cm1[2,2] + cm1[1,2] + cm1[2,1])


#####   Build the model with balanced dataset(over-sampling)  #####
data_balanced_over <- ovun.sample(left ~ ., data = data2, method = "over",N = 22856)$data
table(data_balanced_over$left)

## Split the dataset into training and testing
a2 = seq(1,nrow(data_balanced_over),1)

ind2 = sample(a2, floor(nrow(data_balanced_over)*0.8), replace = FALSE) # randomly select 80% of total records from 
# original dataset without replacement
train2 = data_balanced_over[ind2,]
test2 = data_balanced_over[-ind2,]


#####   Estimation   ######
m2 = glm(left~., data=train2, family = binomial(link='logit'))
summary(m2)

###  Prediction / Evaluation   ###
(m2.logodd = predict(m2, newdata=test2[,-7], type="link"))
(m2.class = as.numeric(m2.logodd>0)) # Convert z-value into class (0,1)
(m2.right = sum(as.numeric(test2$left == m2.class)))
(m2.acc = mean(as.numeric(test2$left == m2.class)))

#####   Build the model with balanced dataset(under-sampling)  #####
data_balanced_under <- ovun.sample(left ~ ., data = data2, method = "under",N = 7142)$data
table(data_balanced_under$left)

## Split the dataset into training and testing
a3 = seq(1,nrow(data_balanced_under),1)

ind3 = sample(a3, floor(nrow(data_balanced_under)*0.8), replace = FALSE) # randomly select 80% of total records from 
# original dataset without replacement
train3 = data_balanced_under[ind3,]
test3 = data_balanced_under[-ind3,]


#####   Estimation   ######
m3 = glm(left~., data=train3, family = binomial(link='logit'))
summary(m3)


###  Prediction / Evaluation   ###
(m3.logodd = predict(m3, newdata=test3[,-7], type="link"))
(m3.class = as.numeric(m3.logodd>0)) # Convert z-value into class (0,1)
(m3.right = sum(as.numeric(test3$left == m3.class)))
(m3.acc = mean(as.numeric(test3$left == m3.class)))


#####   Build the model with balanced dataset(both method)  #####
data_balanced_both <- ovun.sample(left ~ ., data = data2, method = "both", p=0.5, N = 14999)$data
table(data_balanced_both$left)

## Split the dataset into training and testing
a4 = seq(1,nrow(data_balanced_both),1)

ind4 = sample(a4, floor(nrow(data_balanced_both)*0.8), replace = FALSE) # randomly select 80% of total records from 
# original dataset without replacement
train4 = data_balanced_both[ind4,]
test4 = data_balanced_both[-ind4,]


#####   Estimation   ######
m4 = glm(left~., data=train4, family = binomial(link='logit'))
summary(m4)


###  Prediction / Evaluation   ###
(m4.logodd = predict(m4, newdata=test4[,-7], type="link"))
(m4.class = as.numeric(m4.logodd>0)) # Convert z-value into class (0,1)
(m4.right = sum(as.numeric(test4$left == m4.class)))
(m4.acc = mean(as.numeric(test4$left == m4.class)))

#Comparison of different models
AIC(m2,m3,m4)
BIC(m2,m3,m4)


#Building correlation matrix
library(corrplot)
CorTable2 <- cor(data2[,1:19])
corrplot(CorTable2, method = "color", addCoef.col="black", type= "upper", number.cex=0.5)

#Building correlation matrix
library(corrplot)
CorTable1 <- cor(data[,1:10])
corrplot(CorTable1, method = "color", addCoef.col="black", type= "upper", number.cex=0.5)

#Building Decision Tree
library(rpart)
library(rpart.plot)
#Over sampling
#Building Decision Tree Base
decision_tree_over_base <- rpart(left ~., data = train2, method = 'class', 
                                 control = rpart.control(cp=0))
plotcp(decision_tree_over_base)
#rpart.plot(decision_tree_over_base, extra = 104)
rpart.plot(decision_tree_over_base,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_over_base$variable.importance #variable importance
#Building Decision Tree Prepurning
decision_tree_over_pre <- rpart(left ~., data = train2, method = 'class',
                                control = rpart.control(cp=0,maxdepth = 28,
                                                        minsplit = 50))
plotcp(decision_tree_over_pre)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_over_pre,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_over_pre$variable.importance #variable importance

#Building Decision Tree Postpurning
decision_tree_over_post <- rpart(left ~., data = train2, method = 'class',
                                 control = rpart.control(cp=0.00098641,maxdepth = 28,
                                                         minsplit = 50))
plotcp(decision_tree_over_post)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_over_post,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_over_post$variable.importance #variable importance


#Under-sampling
#Building Decision Tree Base
decision_tree_under_base <- rpart(left ~., data = train3, method = 'class', 
                                  control = rpart.control(cp=0))
plotcp(decision_tree_under_base)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_under_base,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_under_base$variable.importance #variable importance
#Building Decision Tree Prepurning
decision_tree_under_pre <- rpart(left ~., data = train3, method = 'class',
                                 control = rpart.control(cp=0,maxdepth = 12,
                                                         minsplit = 50))
plotcp(decision_tree_under_pre)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_under_pre,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_under_pre$variable.importance #variable importance

#Building Decision Tree Postpurning
decision_tree_under_post <- rpart(left ~., data = train3, method = 'class',
                                  control = rpart.control(cp=0.0024536,maxdepth = 12,
                                                          minsplit = 50))
plotcp(decision_tree_under_post)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_under_post,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_under_post$variable.importance #variable importance


#Both sampling
#Building Decision Tree Base
decision_tree_both_base <- rpart(left ~., data = train4, method = 'class', 
                                 control = rpart.control(cp=0))
plotcp(decision_tree_both_base)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_both_base,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_both_base$variable.importance #variable importance
#Building Decision Tree Prepurning
decision_tree_both_pre <- rpart(left ~., data = train4, method = 'class',
                                control = rpart.control(cp=0,maxdepth = 25,
                                                        minsplit = 50))
plotcp(decision_tree_both_pre)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_both_pre,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_both_pre$variable.importance #variable importance

#Building Decision Tree Postpurning
decision_tree_both_post <- rpart(left ~., data = train4, method = 'class',
                                 control = rpart.control(cp=0.00134161,maxdepth = 25,
                                                         minsplit = 50))
plotcp(decision_tree_both_post)
#rpart.plot(decision_tree_over, extra = 104)
rpart.plot(decision_tree_both_post,
           type=0, extra=104, under=FALSE, clip.right.labs=TRUE,
           fallen.leaves=FALSE, 
           digits=2, varlen=-8, faclen=3,
           cex=NULL, tweak=1.2,
           compress=TRUE,
           snip=FALSE)
decision_tree_both_post$variable.importance #variable importance

#Evaluation of Over_base
test2$pred_over_base = predict(decision_tree_over_base,
                               test2, type = 'prob')[,2]
(over_base_accuracy = mean(test2$pred_over_base == test2$left))

#Evaluation of Over_pre
test2$pred_over_pre = predict(decision_tree_over_pre,
                              test2, type = 'prob')[,2]
(over_pre_accuracy = mean(test2$pred_over_pre  == test2$left))

#Evaluation of Over_post
test2$pred_over_post = predict(decision_tree_over_post,
                               test2, type = 'prob')[,2]
(over_post_accuracy = mean(test2$pred_over_post  == test2$left))

(df_acc_over_tree=data.frame(over_base_accuracy,over_pre_accuracy,over_post_accuracy))


#Evaluation / Undersampling /decision_tree_under
#Evaluation of Under_base

test3$pred_under_base = predict(decision_tree_under_base,
                                test3, type = 'prob')[,2]
(under_base_accuracy = mean(test3$pred_under_base == test3$left))

#Evaluation of Under_pre
test3$pred_under_pre = predict(decision_tree_under_pre,
                               test3, type = 'prob')[,2]
(under_pre_accuracy = mean(test3$pred_under_pre  == test3$left))

#Evaluation of Under_post
test3$pred_under_post = predict(decision_tree_under_post,
                                test3, type = 'prob')[,2]
(under_post_accuracy = mean(test3$pred_under_post  == test3$left))

(df_acc_under_tree=data.frame(under_base_accuracy,under_pre_accuracy,under_post_accuracy))

#Evaluation / Both Methods /decision_tree_both
#Evaluation of Both_base

test4$pred_both_base = predict(decision_tree_both_base,
                               test4, type = 'prob')[,2]
(both_base_accuracy = mean(test4$pred_both_base == test4$left))

#Evaluation of Both_pre
test4$pred_both_pre = predict(decision_tree_both_pre,
                              test4, type = 'prob')[,2]
(both_pre_accuracy = mean(test4$pred_both_pre  == test4$left))

#Evaluation of Both_post
test4$pred_both_post = predict(decision_tree_both_post,
                               test4, type = 'prob')[,2]
(both_post_accuracy = mean(test4$pred_both_post  == test4$left))

(df_acc_both_tree=data.frame(both_base_accuracy,both_pre_accuracy,both_post_accuracy))

df_acc_over_tree
df_acc_under_tree
df_acc_both_tree

plotcp(decision_tree_over_base)
printcp(decision_tree_over_base)
plotcp(decision_tree_under_base)
printcp(decision_tree_under_base)
plotcp(decision_tree_both_base)
printcp(decision_tree_both_base)

#### Plot ROC ######
#Logistic

library(ROCR)
prediction_over = prediction(m2.logodd, test2$left)
performace_over = performance(prediction_over, "tpr", "fpr")
prediction_under = prediction(m3.logodd, test3$left)
performace_under = performance(prediction_under, "tpr", "fpr")
prediction_both = prediction(m4.logodd, test4$left)
performace_both = performance(prediction_both, "tpr", "fpr")

plot.new()
plot(performace_over, col= "deeppink")
plot(performace_under, add = TRUE, col= "cyan3")
plot(performace_both, add = TRUE, col= "blueviolet")

abline(0,1, col = "red")
title("ROC curves of Logistic")
legend(0.7, 0.5 ,c("Over", "Under", "Both"), 
       lty = c(1,1,1), 
       lwd = c(0.5,0.5,0.5),
       col = c("deeppink", "cyan3", "blueviolet"),
       ncol=1, cex=0.6, y.intersp=0.5)

#Decision Tree
prediction_over_base = prediction(test2$pred_over_base, test2$left)
performace_over_base = performance(prediction_over_base, "tpr", "fpr")
prediction_over_pre = prediction(test2$pred_over_pre, test2$left)
performace_over_pre = performance(prediction_over_pre, "tpr", "fpr")
prediction_over_post = prediction(test2$pred_over_post, test2$left)
performace_over_post = performance(prediction_over_post, "tpr", "fpr")

prediction_under_base = prediction(test3$pred_under_base, test3$left)
performace_under_base = performance(prediction_under_base, "tpr", "fpr")
prediction_under_pre = prediction(test3$pred_under_pre, test3$left)
performace_under_pre = performance(prediction_under_pre, "tpr", "fpr")
prediction_under_post = prediction(test3$pred_under_post, test3$left)
performace_under_post = performance(prediction_under_post, "tpr", "fpr")

prediction_both_base = prediction(test4$pred_both_base, test4$left)
performace_both_base = performance(prediction_both_base, "tpr", "fpr")
prediction_both_pre = prediction(test4$pred_both_pre, test4$left)
performace_both_pre = performance(prediction_both_pre, "tpr", "fpr")
prediction_both_post = prediction(test4$pred_both_post, test4$left)
performace_both_post = performance(prediction_both_post, "tpr", "fpr")


plot.new()
plot(performace_over_base, col= "deeppink")
plot(performace_over_pre, add = TRUE, col= "cyan3")
plot(performace_over_post, add = TRUE, col= "blueviolet")
plot(performace_under_base, add = TRUE, col= "red" )
plot(performace_under_pre, add = TRUE, col= "yellow")
plot(performace_under_post, add = TRUE, col= "brown")
plot(performace_both_base, add = TRUE, col= "plum")
plot(performace_both_pre, add = TRUE, col= "orange")
plot(performace_both_post, add = TRUE, col= "green")

abline(0,1, col = "red")
title("ROC curves of Decision Tree")
legend(0.7, 0.5 ,c("over_base", "over_pre","over_post",
                   "under_base", "under_pre", "under_post",
                   "both_base", "both_pre", "both_post"), 
       lty = c(1,1,1), 
       lwd = c(0.5,0.5,0.5),
       col = c("deeppink", "cyan3", "blueviolet","red",
               "yellow","brown","plum","orange","green"),
       ncol=1, cex=0.6, y.intersp=0.5)


library(pROC)
par(pty="s") 
OverBaseROC <- roc(test2$left ~ test2$pred_over_base,plot=TRUE,print.auc=TRUE,col="darkgreen",lwd =4,print.auc.y=0.95,legacy.axes=TRUE,main="ROC Curves")
OverPreROC <- roc(test2$left ~ test2$pred_over_pre,plot=TRUE,print.auc=TRUE,col="red",lwd =4,print.auc.y=0.9,legacy.axes=TRUE,add = TRUE)
OverPostROC <- roc(test2$left ~ test2$pred_over_post,plot=TRUE,print.auc=TRUE,col="blue",lwd =4,print.auc.y=0.85,legacy.axes=TRUE,add = TRUE)
UnderBaseROC <- roc(test3$left ~ test3$pred_under_base,plot=TRUE,print.auc=TRUE,col="blueviolet",lwd =4,print.auc.y=0.8,legacy.axes=TRUE,add = TRUE)
UnderPreROC <- roc(test3$left ~ test3$pred_under_pre,plot=TRUE,print.auc=TRUE,col="deeppink",lwd =4,print.auc.y=0.75,legacy.axes=TRUE,add = TRUE)
UnderPostROC <- roc(test3$left ~ test3$pred_under_post,plot=TRUE,print.auc=TRUE,col="plum",lwd =4,print.auc.y=0.7,legacy.axes=TRUE,add = TRUE)
BothBaseROC <- roc(test4$left ~ test4$pred_both_base,plot=TRUE,print.auc=TRUE,col="brown",lwd =4,print.auc.y=0.65,legacy.axes=TRUE,add = TRUE)
BothPreROC <- roc(test4$left ~ test4$pred_both_pre,plot=TRUE,print.auc=TRUE,col="orange",lwd =4,print.auc.y=0.6,legacy.axes=TRUE,add = TRUE)
BothPostROC <- roc(test4$left ~ test4$pred_both_post,plot=TRUE,print.auc=TRUE,col="cyan3",lwd =4,print.auc.y=0.5,legacy.axes=TRUE,add = TRUE)
## Setting levels: control = 0, case = 1
## Setting direction: controls < cases
# svmROC <- roc(vdata_Y ~ svm_predict,plot=TRUE,print.auc=TRUE,col="blue",lwd = 4,print.auc.y=0.4,legacy.axes=TRUE,add = TRUE)
## Setting levels: control = 0, case = 1 ## Setting direction: controls < cases
# legend("topright",inset = c(-2.4,0),legend=c("OverBaseROC","OverPreROC","OverPostROC",
#                          "UnderBaseROC", "UnderPreROC", "UnderPostROC",
#                          "BothBaseROC","BothPreROC","BothPostROC"),pch = 1:9,col=1:9)
legend(0.3,0.45,legend=c("OverBaseROC","OverPreROC","OverPostROC",
                         "UnderBaseROC", "UnderPreROC", "UnderPostROC",
                         "BothBaseROC","BothPreROC","BothPostROC"),col=c("darkgreen", "red",
                                                                         "blue","blueviolet","deeppink",
                                                                         "plum","brown","orange",
                                                                         "cyan3"),lwd=1,
       bty='n',cex=0.6)
legend("topright", inset = c(- 0.4, 0),                   # Create legend outside of plot
       legend = c("Group 1","Group 2"),
       pch = 1:2,
       col = 1:2)
legend("topright", inset=c(-0.2,0), legend=c("A","B"), pch=c(1,3), title="Group")



dt_over_base = decision_tree_over_base$variable.importance
dt_over_pre = decision_tree_over_pre$variable.importance
dt_over_post = decision_tree_over_post$variable.importance
dt_under_base = decision_tree_under_base$variable.importance
dt_under_pre = decision_tree_under_pre$variable.importance
dt_under_post = decision_tree_under_post$variable.importance
dt_both_base = decision_tree_both_base$variable.importance
dt_both_pre = decision_tree_under_pre$variable.importance
dt_both_post =decision_tree_under_post$variable.importance

data.frame(dt_over_base,dt_over_pre,dt_over_post)

dt_under_pre,dt_under_post,dt_both_base,dt_both_pre ,
dt_both_post)

#Variable importance
library(Boruta)
data$left<-as.factor(data$left)
boruta.train <- Boruta(left~., data = data, doTrace = 2)

print(boruta.train)
plot(boruta.train, xlab = "", xaxt = "n")

lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
        boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

# Let's compare the means of our employee turnover satisfaction against the employee population satisfaction
emp_population_satisfaction <-mean(data$satisfaction_level)
left_pop<-subset(data,left==1)
emp_turnover_satisfaction <-mean(left_pop$satisfaction_level)
print( c('The mean for the employee population is: ', emp_population_satisfaction) )
print( c('The mean for the employees that had a turnover is: ' ,emp_turnover_satisfaction) )

# Distribution Plots (Satisfaction - Evaluation - AverageMonthlyHours)
par(mfrow=c(1,3))
hist(data$satisfaction_level, col="green")
hist(data$last_evaluation, col="red")
hist(data$average_montly_hours, col="blue")

#Salary V.S. Turnover
vis_1<-table(data$salary,data$left)
#print(vis_1)
d_vis_1<-as.data.frame(vis_1)
print(d_vis_1)
library(ggplot2)
p<-ggplot(d_vis_1, aes(x=Var1,y=Freq,fill=Var2)) +
        geom_bar(position="dodge",stat='identity') + coord_flip()

print(p)

#Department vsTurnover

vis_2<-table(data$sales,data$left)
d_vis_2<-as.data.frame(vis_2)
d_vis_2<-subset(d_vis_2,Var2==1)
#print(d_vis_2)
library(ggplot2)
d_vis_2$Var1 <- factor(d_vis_2$Var1, levels = d_vis_2$Var1[order(-d_vis_2$Freq)])
p<-ggplot(d_vis_2, aes(x=Var1,y=Freq,fill=Var1)) +
        geom_bar(stat='identity') +theme(axis.text.x = element_text(angle = 90, hjust = 1))

print(p)


#Turnover V.S. ProjectCount

vis_3<-table(data$number_project,data$left)
d_vis_3<-as.data.frame(vis_3)
#print(d_vis_1)
library(ggplot2)
p<-ggplot(d_vis_3, aes(x=Var1,y=Freq,fill=Var2)) +
        geom_bar(position="dodge",stat='identity') + coord_flip()

print(p)


#Turnover V.S. Evaluation

left_data<-subset(data,left==1)
stay_data<-subset(data,left==0)

ggplot() + geom_density(aes(x=last_evaluation), colour="red", data=left_data, size=4) + 
        geom_density(aes(x=last_evaluation), colour="green", data=stay_data, size=4) 

#Turnover V.S. AverageMonthlyHours
ggplot() + geom_density(aes(x=average_montly_hours), colour="red", data=left_data,size=4) + 
        geom_density(aes(x=average_montly_hours), colour="green", data=stay_data, size=4)

#Turnover V.S. Satisfaction
ggplot() + geom_density(aes(x=satisfaction_level), colour="red", data=left_data, size=4) + 
        geom_density(aes(x=satisfaction_level), colour="green", data=stay_data, size=4)

#ProjectCount VS AverageMonthlyHours
library(ggplot2)
p<-ggplot(data, aes(x = factor(number_project), y = average_montly_hours, fill = factor(left))) +
        geom_boxplot() + scale_fill_manual(values = c("green", "red"))
print(p)

#ProjectCount VS Evaluation
p<-ggplot(data, aes(x = factor(number_project), y = last_evaluation, fill = factor(left))) +
        geom_boxplot() + scale_fill_manual(values = c("green", "red"))
print(p)



library(ggplot2)
ggplot(data, aes(satisfaction_level, last_evaluation, color = left)) +
        geom_point(shape = 16, size = 5, show.legend = FALSE) +
        theme_minimal() +
        scale_color_gradient(low = "blue", high = "yellow")



