rm(list = ls())

library(ggplot2)
library(xgboost)
library(glmnet)
library(mlr)

setwd('e:\\roc_calibration')
source("utility.R")

dat <- read.csv('creditcard.csv')
dat <- na.omit(dat)

train.start <- quantile(dat$Time, 0.7)
tune.start <- quantile(dat$Time, 0.7 * 0.7)

dat$Set <- 'Test'
dat$Set[dat$Time <= train.start] <- 'Tune'
dat$Set[dat$Time <= tune.start] <- 'Train'

model.master <- function(data, learners, predictors, random.iter = 1) {

	learners.tags <- learners
	params <- list(0, 0)
	for(i in 1:2) {
		learners.tags[i] <- substr(learners[i], 9, nchar(learners[i]))
		if(learners[i] == 'classif.xgboost') {
			params[[i]] <- params.xgboost
		} else if(learners[i] == 'classif.svm') {
			params[[i]] <- params.svm
		} else if(learners[i] == 'classif.glmnet') {
			params[[i]] <- params.glmnet
		}
	}

	message("Generate different tasks.")
	train.task = makeClassifTask(data = data[data$Set == 'Train', c(predictors, 'Class')], target = "Class")
	train.tune.task = makeClassifTask(data = data[data$Set != 'Test', c(predictors, 'Class')], target = "Class")
	tune.task = makeClassifTask(data = data[data$Set == 'Tune', c(predictors, 'Class')], target = "Class")
	test.task = makeClassifTask(data = data[data$Set == 'Test', c(predictors, 'Class')], target = "Class")

	message("Tuning parameters")
	lrn = myMakeLearner(cl = learners[1], predict.type = "prob")
	ctrl = makeTuneControlRandom(maxit = random.iter)
	rdesc = makeFixedHoldoutInstance(train.inds = which(data$Set[data$Set != 'Test'] == 'Train'), test.inds = which(data$Set[data$Set != 'Test'] == 'Tune'), size = nrow(data[data$Set != 'Test', ]))
	opts = tuneParams(learner = lrn, task = train.tune.task, resampling = rdesc, par.set = params[[1]], control = ctrl)
	message("Optimal tuning parameters: ", paste(opts$x, collapse = " "))

	message("Train the model and make predictions.")
	lrn = setHyperPars(lrn, par.vals = opts$x)
	message("Train the model based on training data.")
	model.train <- train(lrn, train.task)
	message("Train the model jointly on train+tune data.")
	model.train.tune <- train(lrn, train.tune.task)

	pred.tune.model1 <- as.data.frame(predict(model.train, task = tune.task))$prob.1
	pred.test.model1 <- as.data.frame(predict(model.train.tune, task = test.task))$prob.1

	message("Calculate ROC curves.")
	roc.tune <- gamma.func(pred.tune.model1, data$Class[data$Set == 'Tune'])
	pred.test.cal.model1 <- caliberate.roc(pred.test.model1, roc.tune)

	plotdat <- data.frame(
		fpr = c(roc.tune$fpr, roc.tune$x),
		tpr = c(roc.tune$tpr, roc.tune$y),
		type = rep(c('raw', 'smooth'), rep(nrow(roc.tune), 2)),
		method = 'tune')

	print(p1 <- ggplot(data = plotdat, aes(x = fpr, y = tpr)) + geom_line(aes(color = method, linetype = type)))

	message("Fit an independent model.")
	if(learners[2] == learners[1]) {
		model.train <- train(lrn, train.task)
		model.train.tune <- train(lrn, train.tune.task)
	} else {
		lrn = myMakeLearner(cl = learners[2], predict.type = "prob")
		ctrl = makeTuneControlRandom(maxit = random.iter)
		rdesc = makeFixedHoldoutInstance(train.inds = which(data$Set[data$Set != 'Test'] == 'Train'), test.inds = which(data$Set[data$Set != 'Test'] == 'Tune'), size = nrow(data[data$Set != 'Test', ]))
		opts = tuneParams(learner = lrn, task = train.tune.task, resampling = rdesc, par.set = params[[2]], control = ctrl)
		lrn = setHyperPars(lrn, par.vals = opts$x)
		model.train <- train(lrn, train.task)
		model.train.tune <- train(lrn, train.tune.task)
	}

	pred.tune.model2 <- as.data.frame(predict(model.train, task = tune.task))$prob.1
	pred.test.model2 <- as.data.frame(predict(model.train.tune, task = test.task))$prob.1

	roc.tune <- gamma.func(pred.tune.model2, data$Class[data$Set == 'Tune'])
	pred.test.cal.model2 <- caliberate.roc(pred.test.model2, roc.tune)

	pred.test.12.cal <- pred.test.cal.model1 * pred.test.model2 / (1 - pred.test.model2 + 1e-10)
	pred.test.21.cal <- pred.test.cal.model2 * pred.test.model1 / (1 - pred.test.model1 + 1e-10)

	roc.test.model1 <- roc(pred.test.model1, data$Class[data$Set == 'Test'])
	roc.test.model2 <- roc(pred.test.model2, data$Class[data$Set == 'Test'])
	roc.test.12 <- roc(pred.test.12.cal, data$Class[data$Set == 'Test'])
	roc.test.21 <- roc(pred.test.21.cal, data$Class[data$Set == 'Test'])

	auc.model1 <- auc(data$Class[data$Set == 'Test'], pred.test.model1)
	auc.model2 <- auc(data$Class[data$Set == 'Test'], pred.test.model2)
	auc.12 <- auc(data$Class[data$Set == 'Test'], pred.test.12.cal)
	auc.21 <- auc(data$Class[data$Set == 'Test'], pred.test.21.cal)

	message("Correlation between two models ", cor(pred.test.model1, pred.test.model2))

	plotdat <- data.frame(rbind(roc.test.model1, roc.test.model2, roc.test.12, roc.test.21),
					 method = rep(c(paste('caliberated 1, AUC =', round(auc.12, 3)), paste('caliberated 2, AUC =', round(auc.21, 3)),
							paste("Model 1:", learners.tags[1], ", AUC =", round(auc.model1, 3)),
							paste("Model 2:", learners.tags[2], ", AUC =", round(auc.model2, 3))),
					c(nrow(roc.test.model1), nrow(roc.test.model2), nrow(roc.test.12), nrow(roc.test.21))))

	print(p2 <- ggplot(data = plotdat, aes(x = fpr, y = tpr)) + geom_line(aes(linetype = method), cex = 0.2)+ guides(linetype=guide_legend(nrow=4)) + xlab("False Positive Rate") + ylab("True Positive Rate") + theme(legend.position = 'top'))

	return(list(p = p2, roc.data = plotdat))
}

params.xgboost <- makeParamSet(
	makeNumericParam("eta", lower = 0.01, upper = 0.1),
	makeDiscreteParam("max_depth", values = seq(2, 8)),
	makeDiscreteParam("nrounds", values = seq(10, 500))
	)
params.glmnet <- makeParamSet(
	makeNumericParam("alpha", lower = 0.01, upper = 0.99)#,
#	makeNumericParam("lambda", lower = 0., upper = 0.01)
	)
params[[2]] <- makeParamSet(
	makeNumericParam("alpha", lower = 0.01, upper = 0.99)#,
#	makeNumericParam("lambda", lower = 0., upper = 1e-10)
	)
params.svm <- makeParamSet(
	makeNumericParam("cost", lower = 0, upper = 10, trafo = function(x){4^x})
	)

learners <- c('classif.xgboost', 'classif.glmnet')

predictors <- names(dat)[!names(dat) %in% c('Set', 'Class', 'Time')]

for(l1 in c('xgboost', 'glmnet', 'svm')) {
	for(l2 in c('xgboost', 'glmnet', 'svm')) {
		roc_adjust <- model.master(data = dat, learner = c(paste('classif', l1, sep = '.'), paste('classif', l2, sep = '.')), predictors = predictors, random.iter = 10)
		jpeg(paste('experiment3_', l1, '_', l2, '.jpg', sep = ''), width = 4, height = 5, units = 'in', res = 1000)
		print(roc_adjust$p)
		dev.off()
	}
}

roc_adjust$p3

print(roc_adjust$p3)
print(roc_adjust$p2)
print(roc_adjust$p3)
