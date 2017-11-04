rm(list = ls())

library(ggplot2)
library(glmnet)
library(mlr)
library(pROC)
library(unbalanced)
library(DMwR)
library(xgboost)

setwd('e:\\roc_calibration')
source("utility.R")

dat <- read.csv('creditcard.csv')
dat <- na.omit(dat)

train.start <- quantile(dat$Time, 0.7)
tune.start <- quantile(dat$Time, 0.7 * 0.7)

dat$Set <- 'Test'
dat$Set[dat$Time <= train.start] <- 'Tune'
dat$Set[dat$Time <= tune.start] <- 'Train'

predictors <- names(dat)[!names(dat) %in% c('Amount', 'Set', 'Class', 'Time', 'Group')]

params.xgboost <- makeParamSet(
    makeNumericParam("eta", lower = 0.01, upper = 0.1),
    makeDiscreteParam("max_depth", values = seq(2, 8)),
    makeDiscreteParam("nrounds", values = seq(10, 500))
    )
params.glmnet <- makeParamSet(
    makeNumericParam("alpha", lower = 0.01, upper = 0.99),
    makeNumericParam("lambda", lower = 0., upper = 2., trafo = function(x){2^x - 1})
    )
params.svm <- makeParamSet(
    makeNumericParam("cost", lower = 0, upper = 10, trafo = function(x){4^x})
    )

if(FALSE) {
learner = 'classif.xgboost'
params = params.xgboost
random.iter <- 10
method <- 'Weight'
data <- dat
}

dat$Group <- as.numeric(dat$V12 < 0)

for(method in c('Weight', 'ubUnder', 'ubSMOTE')) {
	if(method == 'Weight') {
		method_tag <- 'Weighted'
	} else if (method == 'ubUnder') {
		method_tag <- 'Under-sampling'
	} else {
		method_tag <- 'SMOTE'
	}
	for(learner in c('classif.xgboost', 'classif.glmnet')) {
		if(learner == 'classif.xgboost') {
			learner_tag <- 'xgboost'
			params <- params.xgboost
		} else if (learner == 'classif.svm') {
			learner_tag <- 'SVM'
			params <- params.svm
		} else {
			learner_tag <- 'glmnet'
			params <- params.glmnet
		}
		mod <- model.master(data = dat, method = method, predictors = predictors, learner = learner, params = params, random.iter = 10)
		gc()
		jpeg(paste("figures/experiment1_", method, "_", learner_tag, ".jpg", sep = ""), height = 4, width = 4, units = 'in', res = 1000)
		print(mod$p + ggtitle(paste(method_tag, ", ", learner_tag, sep = "")))
		dev.off()
	}
}

n.simulations <- 50
results <- c()
for(method in c('Weight', 'ubUnder', 'ubSMOTE')) {
	auc.xgboost <- auc.glmnet <- auc.svm <- c(0, 0)
	for(i in seq(n.simulations)) {
		mod.xgboost <- model.master(data = dat, method = method, predictors = predictors, learner = 'classif.xgboost', params = params.xgboost, random.iter = 10)
		gc()
		mod.glmnet <- model.master(data = dat, method = method, predictors = predictors, learner = 'classif.glmnet', params = params.glmnet, random.iter = 10)
		gc()
		mod.svm <- model.master(data = dat, method = method, predictors = predictors, learner = 'classif.svm', params = params.svm, random.iter = 10)
		gc()
		auc.xgboost <- auc.xgboost + mod.xgboost$auc
		auc.svm <- auc.svm  + mod.svm$auc
		auc.glmnet <- auc.glmnet + mod.glmnet$auc
	}
	auc.xgboost <- auc.xgboost / n.simulations
	auc.svm <- auc.svm / n.simulations
	auc.glmnet <- auc.glmnet / n.simulations
	results <- rbind(results, auc.xgboost, auc.svm, auc.glmnet)
}

write.table(round(rbind(c(t(results[c(1, 4, 7), ])),
				c(t(results[c(2, 5, 8), ])),
				c(t(results[c(3, 6, 9), ]))), 3),
		file = 'simulation_results_latex.txt', sep = ' & ')

write.table(round(results, 3), file = 'simulation_results.txt')

# Semi-supervised Learning

mod.pca <- prcomp(dat[, predictors], center = TRUE, scale. = TRUE)
dat$PC1 <- mod.pca$x[, 1]

ggplot(data = dat, aes(x = PC1, y = ..density..)) + geom_histogram() + facet_grid(. ~ Set)

unsupervised.pca <- function(data, predictors, learner, params, random.iter = 5) {
	data$PC1 <- prcomp(data[, predictors], center = TRUE, scale. = TRUE)$x[, 1]
	data$Group <- as.integer(data$PC1 > 0)
	mod.pca <- model.master(data = data, method = 'None', predictors = predictors, learner = learner, params = params, random.iter = random.iter)
	dat$Group <- 0
	mod.raw <- model.master(data = data, method = 'None', predictors = predictors, learner = learner, params = params, random.iter = random.iter)

	roc.pca <- roc(mod.pca$data$cal.scores, mod.pca$data$true.labels)
	roc.raw <- roc(mod.raw$data$raw.scores, mod.raw$data$true.labels)
	auc.pca <- auc(mod.pca$data$true.labels, mod.pca$data$cal.scores)
	auc.raw <- auc(mod.raw$data$true.labels, mod.raw$data$raw.scores)
	plotdat <- data.frame(rbind(roc.pca, roc.raw),
     	                 method = rep(c(paste('Caliberated with PC1, AUC =', round(auc.pca, 3)), paste('Raw, AUC =', round(auc.raw, 3))), c(nrow(roc.pca), nrow(roc.raw))))
	print(p <- ggplot(data = plotdat, aes(x = fpr, y = tpr)) + geom_line(aes(linetype = method), cex = 0.2) + guides(color=guide_legend(nrow=2)) + xlab("False Positive Rate") + ylab("True Positive Rate") + theme(legend.position = 'top'))
	return(list(p = p, model.pca = mod.pca, model.raw = mod.raw))
}

params.glmnet <- makeParamSet(
	makeNumericParam("alpha", lower = 0.01, upper = 0.99),
	makeNumericParam("lambda", lower = 0., upper = 0.1, trafo = function(x) {exp(x) - 1})
)
pca.result.xgboost <- unsupervised.pca(dat, predictors = predictors, learner = 'classif.xgboost', params = params.xgboost, random.iter = 10)
pca.result.glmnet <- unsupervised.pca(dat, predictors = predictors, learner = 'classif.glmnet', params = params.glmnet, random.iter = 10)
pca.result.svm <- unsupervised.pca(dat, predictors = predictors, learner = 'classif.svm', params = params.svm, random.iter = 10)

jpeg("figures/experiment2_xgboost.jpg", height = 4, width = 4, units = 'in', res = 1000)
print(pca.result.xgboost$p + guides(linetype=guide_legend(nrow=2)) + ggtitle("xgboost"))
dev.off()
jpeg("figures/experiment2_svm.jpg", height = 4, width = 4, units = 'in', res = 1000)
print(pca.result.svm$p + guides(linetype=guide_legend(nrow=2)) + ggtitle("SVM"))
dev.off()
jpeg("figures/experiment2_glmnet.jpg", height = 4, width = 4, units = 'in', res = 1000)
print(pca.result.glmnet$p + guides(linetype=guide_legend(nrow=2)) + ggtitle("glmnet"))
dev.off()
